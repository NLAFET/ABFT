/*
 * Copyright (c) 2010-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/sys/atomic.h"

#include "parsec/utils/mca_param.h"
#include "parsec/constants.h"

#if defined(PARSEC_HAVE_CUDA)
#include "parsec/runtime.h"
#include "parsec/data_internal.h"
#include "parsec/devices/cuda/dev_cuda.h"
#include "parsec/profiling.h"
#include "parsec/execution_stream.h"
#include "parsec/arena.h"
#include "parsec/scheduling.h"
#include "parsec/utils/debug.h"
#include "parsec/utils/argv.h"
#include "parsec/utils/zone_malloc.h"
#include "parsec/class/fifo.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#if defined(PARSEC_PROF_TRACE)
/* Accepted values are: PARSEC_PROFILE_CUDA_TRACK_DATA_IN | PARSEC_PROFILE_CUDA_TRACK_DATA_OUT |
 *                      PARSEC_PROFILE_CUDA_TRACK_OWN | PARSEC_PROFILE_CUDA_TRACK_EXEC
 */
int parsec_cuda_trackable_events = PARSEC_PROFILE_CUDA_TRACK_EXEC | PARSEC_PROFILE_CUDA_TRACK_DATA_OUT
    | PARSEC_PROFILE_CUDA_TRACK_DATA_IN | PARSEC_PROFILE_CUDA_TRACK_OWN;
int parsec_cuda_movein_key_start;
int parsec_cuda_movein_key_end;
int parsec_cuda_moveout_key_start;
int parsec_cuda_moveout_key_end;
int parsec_cuda_own_GPU_key_start;
int parsec_cuda_own_GPU_key_end;
#endif  /* defined(PROFILING) */

int parsec_cuda_output_stream = -1;
int32_t parsec_CUDA_sort_pending_list = 0;

extern int32_t parsec_CUDA_d2h_max_flows;
static char* cuda_lib_path = NULL;

static int
parsec_cuda_memory_reserve( gpu_device_t* gpu_device,
                           int           memory_percentage,
                           int           number_of_elements,
                           size_t        eltsize );
static int
parsec_cuda_memory_release( gpu_device_t* gpu_device );

static int cuda_legal_compute_capabilitites[] = {10, 11, 12, 13, 20, 21, 30, 32, 35, 37, 50, 52, 53, 60, 61, 62, 70};

/* look up how many FMA per cycle in single/double, per cuda MP
 * precision.
 * The following table provides updated values for future archs
 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions
 */
static int parsec_cuda_device_lookup_cudamp_floprate(int major, int minor, int *drate, int *srate, int *trate, int *hrate) {
    if( major == 1 ) {
        *hrate = *trate = *srate = 8;
        *drate = 1;
    } else if (major == 2 && minor == 0) {
        *hrate = *trate = *srate = 32;
        *drate = 16;
    } else if (major == 2 && minor == 1) {
        *hrate = *trate = *srate = 48;
        *drate = 4;
    } else if ((major == 3 && minor == 0) ||
               (major == 3 && minor == 2)) {
        *hrate = *trate = *srate = 192;
        *drate = 8;
    } else if ((major == 3 && minor == 5) ||
               (major == 3 && minor == 7)) {
        *hrate = *trate = *srate = 192;
        *drate = 64;
    } else if ((major == 5 && minor == 0) ||
               (major == 5 && minor == 2)) {
        *hrate = *trate = *srate = 128;
        *drate = 4;
    } else if (major == 5 && minor == 3) {
        *hrate = 256;
        *trate = *srate = 128;
        *drate = 4;
    } else if (major == 6 && minor == 0) {
        *hrate = 128;
        *trate = *srate = 64;
        *drate = 32;
    } else if (major == 6 && minor == 1) {
        *hrate = 2;
        *trate = *srate = 128;
        *drate = 4;
    } else if (major == 6 && minor == 2) {
        *hrate = 256;
        *trate = *srate = 128;
        *drate = 4;
    } else if (major == 7 && minor == 0) {
        *hrate = 128;
        *trate = 512;
        *srate = 64;
        *drate = 32;
    } else {
        parsec_debug_verbose(3, parsec_cuda_output_stream, "Unsupported GPU %d, %d, skipping.", major, minor);
        return PARSEC_ERROR;
    }
    return PARSEC_SUCCESS;
}


static int parsec_cuda_device_fini(parsec_device_t* device)
{
    gpu_device_t* gpu_device = (gpu_device_t*)device;
    cudaError_t status;
    int j, k;

    status = cudaSetDevice( gpu_device->cuda_index );
    PARSEC_CUDA_CHECK_ERROR( "(parsec_cuda_device_fini) cudaSetDevice ", status,
                            {continue;} );

    /* Release the registered memory */
    parsec_cuda_memory_release(gpu_device);

    /* Release pending queue */
    OBJ_DESTRUCT(&gpu_device->pending);

    /* Release all streams */
    for( j = 0; j < gpu_device->max_exec_streams; j++ ) {
        parsec_gpu_exec_stream_t* exec_stream = &(gpu_device->exec_stream[j]);

        exec_stream->executed = 0;
        exec_stream->start    = 0;
        exec_stream->end      = 0;

        for( k = 0; k < exec_stream->max_events; k++ ) {
            assert( NULL == exec_stream->tasks[k] );
            status = cudaEventDestroy(exec_stream->events[k]);
            PARSEC_CUDA_CHECK_ERROR( "(parsec_cuda_device_fini) cudaEventDestroy ", status,
                                    {continue;} );
        }
        exec_stream->max_events = 0;
        free(exec_stream->events); exec_stream->events = NULL;
        free(exec_stream->tasks); exec_stream->tasks = NULL;
        free(exec_stream->fifo_pending); exec_stream->fifo_pending = NULL;
        /* Release the stream */
        cudaStreamDestroy( exec_stream->cuda_stream );
        free(exec_stream->name);
    }
    free(gpu_device->exec_stream); gpu_device->exec_stream = NULL;

    gpu_device->cuda_index = -1;

    /* Cleanup the GPU memory. */
    OBJ_DESTRUCT(&gpu_device->gpu_mem_lru);
    OBJ_DESTRUCT(&gpu_device->gpu_mem_owned_lru);

    return PARSEC_SUCCESS;
}

static int parsec_cuda_memory_register(parsec_device_t* device, parsec_data_collection_t* desc, void* ptr, size_t length)
{
    cudaError_t status;
    int rc = PARSEC_ERROR;

    /* Memory needs to be registered only once with CUDA. */
    if (desc->memory_registration_status == MEMORY_STATUS_REGISTERED) {
        rc = PARSEC_SUCCESS;
        return rc;
    }

    /*
     * We rely on the thread-safety of the CUDA interface to register the memory
     * as another thread might be submiting tasks at the same time
     * (cuda_scheduling.h), and we do not set a device since we register it for
     * all devices.
     */
    status = cudaHostRegister(ptr, length, cudaHostRegisterPortable );
    PARSEC_CUDA_CHECK_ERROR( "(parsec_cuda_memory_register) cudaHostRegister ", status,
                            { goto restore_and_return; } );

    rc = PARSEC_SUCCESS;
    desc->memory_registration_status = MEMORY_STATUS_REGISTERED;

  restore_and_return:
    (void)device;
    return rc;
}

static int parsec_cuda_memory_unregister(parsec_device_t* device, parsec_data_collection_t* desc, void* ptr)
{
    cudaError_t status;
    int rc = PARSEC_ERROR;

    /* Memory needs to be registered only once with CUDA. One registration = one deregistration */
    if (desc->memory_registration_status == MEMORY_STATUS_UNREGISTERED) {
        rc = PARSEC_SUCCESS;
        return rc;
    }

    /*
     * We rely on the thread-safety of the CUDA interface to unregister the memory
     * as another thread might be submiting tasks at the same time (cuda_scheduling.h)
     */
    status = cudaHostUnregister(ptr);
    PARSEC_CUDA_CHECK_ERROR( "(parsec_cuda_memory_ununregister) cudaHostUnregister ", status,
                            {continue;} );

    rc = PARSEC_SUCCESS;
    desc->memory_registration_status = MEMORY_STATUS_UNREGISTERED;

    (void)device;
    return rc;
}


void* cuda_find_incarnation(gpu_device_t* gpu_device,
                            const char* fname)
{
    char library_name[FILENAME_MAX], function_name[FILENAME_MAX], *env;
    int i, index, capability = gpu_device->major * 10 + gpu_device->minor;
    cudaError_t status;
    void *fn = NULL;
    char** argv = NULL;

    status = cudaSetDevice( gpu_device->cuda_index );
    PARSEC_CUDA_CHECK_ERROR( "(cuda_find_incarnation) cudaSetDevice ", status, {continue;} );

    for( i = 0, index = -1; i < (int)(sizeof(cuda_legal_compute_capabilitites)/sizeof(int)); i++ ) {
        if(cuda_legal_compute_capabilitites[i] == capability) {
            index = i;
            break;
        }
    }
    if( -1 == index ) {  /* This shouldn't have happened */
        return NULL;
    }

    /**
     * Prepare the list of PATH or FILE to be searched for a CUDA shared library.
     * In any case this list might be a list of ; separated possible targets,
     * where each target can be either a directory or a specific file.
     */
    env = getenv("PARSEC_CUCORES_LIB");
    if( NULL != env ) {
        argv = parsec_argv_split(env, ';');
    } else if( NULL != cuda_lib_path ) {
        argv = parsec_argv_split(cuda_lib_path, ';');
    }

  retry_lesser_sm_version:
    if( -1 == index ) {
        capability = 0;
        snprintf(function_name, FILENAME_MAX, "%s", fname);
    }
    else {
        capability = cuda_legal_compute_capabilitites[index];
        snprintf(function_name, FILENAME_MAX, "%s_SM%2d", fname, capability);
    }

    if( capability )
        snprintf(library_name,  FILENAME_MAX, "libdplasma_cucores_sm%d.so", capability);
    else
        snprintf(library_name,  FILENAME_MAX, "libdplasma_cores_cuda.so");

    fn = parsec_device_find_function(function_name, library_name, (const char**)argv);
    if( NULL == fn ) {  /* look for the function with lesser capabilities */
        parsec_debug_verbose(10, parsec_cuda_output_stream,
                            "No function %s found for CUDA device %s",
                            function_name, gpu_device->super.name);
        index--;
        if(-1 <= index)
            goto retry_lesser_sm_version;
    }

    if( NULL != argv )
        parsec_argv_free(argv);

    return fn;
}

/**
 * Register a taskpool with a device by checking that the device
 * supports the dynamic function required by the different incarnations.
 * If multiple devices of the same type exists we assume thay all have
 * the same capabilities.
 */
static int
parsec_cuda_taskpool_register(parsec_device_t* device,
                              parsec_taskpool_t* tp)
{
    gpu_device_t* gpu_device = (gpu_device_t*)device;
    int32_t rc = PARSEC_ERR_NOT_FOUND;
    uint32_t i, j;

    /**
     * Detect if a particular chore has a dynamic load dependency and if yes
     * load the corresponding module and find the function.
     */
    assert(PARSEC_DEV_CUDA == device->type);
    assert(tp->devices_index_mask & (1 << device->device_index));

    for( i = 0; i < tp->nb_task_classes; i++ ) {
        const parsec_task_class_t* tc = tp->task_classes_array[i];
        __parsec_chore_t* chores = (__parsec_chore_t*)tc->incarnations;
        for( j = 0; NULL != chores[j].hook; j++ ) {
            if( chores[j].type != device->type )
                continue;
            if( NULL != chores[j].dyld_fn ) {
                /* the function has been set for another device of the same type */
                return PARSEC_SUCCESS;
            }
            if ( NULL == chores[j].dyld ) {
                chores[j].dyld_fn = NULL;  /* No dynamic support required for this kernel */
                rc = PARSEC_SUCCESS;
            } else {
                void* devf = cuda_find_incarnation(gpu_device, chores[j].dyld);
                if( NULL != devf ) {
                    chores[j].dyld_fn = devf;
                    rc = PARSEC_SUCCESS;
                }
            }
        }
    }
    if( PARSEC_SUCCESS != rc ) {
        tp->devices_index_mask &= ~(1 << device->device_index);  /* drop support for this device */
        parsec_debug_verbose(10, parsec_cuda_output_stream,
                             "Device %d (%s) disabled for taskpool %p", device->device_index, device->name, tp);
    }
    return rc;
}

static int
parsec_cuda_taskpool_unregister(parsec_device_t* device, parsec_taskpool_t* tp)
{
    (void)device; (void)tp;
    return PARSEC_SUCCESS;
}

int parsec_gpu_init(parsec_context_t *parsec_context)
{
    int cuda_memory_block_size, cuda_memory_percentage, cuda_memory_number_of_blocks = -1;
    int show_caps_index, show_caps = 0;
    int use_cuda_index, use_cuda;
    int cuda_mask, cuda_verbosity;
    int ndevices, i, j, k;
    cudaError_t cudastatus;

    use_cuda_index = parsec_mca_param_reg_int_name("device_cuda", "enabled",
                                                   "The number of CUDA device to enable for the next PaRSEC context (-1 for all available)",
                                                   false, false, -1, &use_cuda);
    (void)parsec_mca_param_reg_int_name("device_cuda", "mask",
                                        "The bitwise mask of CUDA devices to be enabled (default all)",
                                        false, false, 0xffffffff, &cuda_mask);
    (void)parsec_mca_param_reg_int_name("device_cuda", "verbose",
                                        "Set the verbosity level of the CUDA device (negative value: use debug verbosity), higher is less verbose)\n",
                                        false, false, -1, &cuda_verbosity);
    (void)parsec_mca_param_reg_string_name("device_cuda", "path",
                                           "Path to the shared library files containing the CUDA version of the hooks. It is a ;-separated list of either directories or .so files.\n",
                                           false, false, PARSEC_LIB_CUDA_PREFIX, &cuda_lib_path);
    (void)parsec_mca_param_reg_int_name("device_cuda", "memory_block_size",
                                        "The CUDA memory page for PaRSEC internal management (in bytes).",
                                        false, false, 512*1024, &cuda_memory_block_size);
    (void)parsec_mca_param_reg_int_name("device_cuda", "memory_use",
                                        "The percentage of the total GPU memory to be used by this PaRSEC context",
                                        false, false, 95, &cuda_memory_percentage);
    (void)parsec_mca_param_reg_int_name("device_cuda", "memory_number_of_blocks",
                                        "Alternative to device_cuda_memory_use: sets exactly the number of blocks to allocate (-1 means to use a percentage of the available memory)",
                                        false, false, -1, &cuda_memory_number_of_blocks);
    (void)parsec_mca_param_reg_int_name("device_cuda", "max_number_of_ejected_data",
                                        "Sets up the maximum number of blocks that can be ejected from GPU memory",
                                        false, false, MAX_PARAM_COUNT, &parsec_CUDA_d2h_max_flows);
    (void)parsec_mca_param_reg_int_name("device_cuda", "sort_pending_tasks",
                                        "Boolean to let the GPU engine sort the first pending tasks stored in the list",
                                        false, false, 0, &parsec_CUDA_sort_pending_list);

    if( 0 == use_cuda ) {
        return -1;  /* Nothing to do around here */
    }

    if( cuda_verbosity >= 0 ) {
        parsec_cuda_output_stream = parsec_output_open(NULL);
        parsec_output_set_verbosity(parsec_cuda_output_stream, cuda_verbosity);
    } else {
        parsec_cuda_output_stream = parsec_device_output;
    }

    cudastatus = cudaGetDeviceCount( &ndevices );
    PARSEC_CUDA_CHECK_ERROR( "cudaGetDeviceCount ", cudastatus,
                             {
                                 parsec_mca_param_set_int(use_cuda_index, 0);
                                 return -1;
                             } );

    if( ndevices > use_cuda ) {
        if( 0 < use_cuda_index ) {
            ndevices = use_cuda;
        }
    } else if (ndevices < use_cuda ) {
        if( 0 < use_cuda_index ) {
            parsec_warning("User requested %d GPUs, but only %d are available in this machine. PaRSEC will enable all of them.", use_cuda, ndevices);
            parsec_mca_param_set_int(use_cuda_index, ndevices);
        }
    }

    /* Update the number of GPU for the upper layer */
    use_cuda = ndevices;
    if( 0 == ndevices ) {
        return -1;
    }

    show_caps_index = parsec_mca_param_find("device", NULL, "show_capabilities");
    if(0 < show_caps_index) {
        parsec_mca_param_lookup_int(show_caps_index, &show_caps);
    }
#if defined(PARSEC_PROF_TRACE)
    parsec_profiling_add_dictionary_keyword( "movein", "fill:#33FF33",
                                             sizeof(intptr_t), "pointer{int64_t}",
                                             &parsec_cuda_movein_key_start, &parsec_cuda_movein_key_end);
    parsec_profiling_add_dictionary_keyword( "moveout", "fill:#ffff66",
                                             sizeof(intptr_t), "pointer{int64_t}",
                                             &parsec_cuda_moveout_key_start, &parsec_cuda_moveout_key_end);
    parsec_profiling_add_dictionary_keyword( "cuda", "fill:#66ff66",
                                             0, NULL,
                                             &parsec_cuda_own_GPU_key_start, &parsec_cuda_own_GPU_key_end);
#endif  /* defined(PROFILING) */

    for( i = 0; i < ndevices; i++ ) {
        gpu_device_t* gpu_device;
        char *szName;
        int major, minor, concurrency, computemode, streaming_multiprocessor, drate, srate, trate, hrate;
        float clockRate;
        struct cudaDeviceProp prop;

        /* Allow fine grain selection of the GPU's */
        if( !((1 << i) & cuda_mask) ) continue;

        cudastatus = cudaSetDevice( i );
        PARSEC_CUDA_CHECK_ERROR( "cudaSetDevice ", cudastatus, {continue;} );
        cudastatus = cudaGetDeviceProperties( &prop, i );
        PARSEC_CUDA_CHECK_ERROR( "cudaGetDeviceProperties ", cudastatus, {continue;} );

        szName    = prop.name;
        major     = prop.major;
        minor     = prop.minor;
        clockRate = prop.clockRate/1e3f;
        concurrency = prop.concurrentKernels;
        streaming_multiprocessor = prop.multiProcessorCount;
        computemode = prop.computeMode;

        gpu_device = (gpu_device_t*)calloc(1, sizeof(gpu_device_t));
        OBJ_CONSTRUCT(gpu_device, parsec_list_item_t);
        gpu_device->cuda_index = (uint8_t)i;
        gpu_device->major      = (uint8_t)major;
        gpu_device->minor      = (uint8_t)minor;
        gpu_device->super.name = strdup(szName);
        gpu_device->data_avail_epoch = 0;

        gpu_device->max_exec_streams = PARSEC_MAX_STREAMS;
        gpu_device->exec_stream =
            (parsec_gpu_exec_stream_t*)malloc(gpu_device->max_exec_streams
                                              * sizeof(parsec_gpu_exec_stream_t));
        for( j = 0; j < gpu_device->max_exec_streams; j++ ) {
            parsec_gpu_exec_stream_t* exec_stream = &(gpu_device->exec_stream[j]);

            /* Allocate the stream */
            cudastatus = cudaStreamCreate( &(exec_stream->cuda_stream) );
            PARSEC_CUDA_CHECK_ERROR( "cudaStreamCreate ", cudastatus,
                                     {break;} );
            exec_stream->workspace    = NULL;
            exec_stream->max_events   = PARSEC_MAX_EVENTS_PER_STREAM;
            exec_stream->executed     = 0;
            exec_stream->start        = 0;
            exec_stream->end          = 0;
            exec_stream->name         = NULL;
            exec_stream->fifo_pending = (parsec_list_t*)OBJ_NEW(parsec_list_t);
            OBJ_CONSTRUCT(exec_stream->fifo_pending, parsec_list_t);
            exec_stream->tasks    = (parsec_gpu_task_t**)malloc(exec_stream->max_events
                                                                * sizeof(parsec_gpu_task_t*));
            exec_stream->events   = (cudaEvent_t*)malloc(exec_stream->max_events * sizeof(cudaEvent_t));
            /* and the corresponding events */
            for( k = 0; k < exec_stream->max_events; k++ ) {
                exec_stream->events[k]   = NULL;
                exec_stream->tasks[k]    = NULL;
                cudastatus = cudaEventCreate(&(exec_stream->events[k]));
                PARSEC_CUDA_CHECK_ERROR( "(INIT) cudaEventCreate ", (cudaError_t)cudastatus,
                                         {break;} );
            }
            if(j == 0) {
                asprintf(&exec_stream->name, "h2d(%d)", j);
            } else if(j == 1) {
                asprintf(&exec_stream->name, "d2h(%d)", j);
            } else {
                asprintf(&exec_stream->name, "cuda(%d)", j);
            }
#if defined(PARSEC_PROF_TRACE)
            exec_stream->profiling = parsec_profiling_thread_init( 2*1024*1024, PARSEC_PROFILE_STREAM_STR, i, j );
            if(j == 0) {
                exec_stream->prof_event_track_enable = parsec_cuda_trackable_events & PARSEC_PROFILE_CUDA_TRACK_DATA_IN;
                exec_stream->prof_event_key_start    = parsec_cuda_movein_key_start;
                exec_stream->prof_event_key_end      = parsec_cuda_movein_key_end;
            } else if(j == 1) {
                exec_stream->prof_event_track_enable = parsec_cuda_trackable_events & PARSEC_PROFILE_CUDA_TRACK_DATA_OUT;
                exec_stream->prof_event_key_start    = parsec_cuda_moveout_key_start;
                exec_stream->prof_event_key_end      = parsec_cuda_moveout_key_end;
            } else {
                exec_stream->prof_event_track_enable = parsec_cuda_trackable_events & PARSEC_PROFILE_CUDA_TRACK_EXEC;
                exec_stream->prof_event_key_start    = -1;
                exec_stream->prof_event_key_end      = -1;
            }
#endif  /* defined(PARSEC_PROF_TRACE) */
        }

        gpu_device->super.type                 = PARSEC_DEV_CUDA;
        gpu_device->super.executed_tasks       = 0;
        gpu_device->super.transferred_data_in  = 0;
        gpu_device->super.transferred_data_out = 0;
        gpu_device->super.required_data_in     = 0;
        gpu_device->super.required_data_out    = 0;

        gpu_device->super.device_fini                = parsec_cuda_device_fini;
        gpu_device->super.device_memory_register     = parsec_cuda_memory_register;
        gpu_device->super.device_memory_unregister   = parsec_cuda_memory_unregister;
        gpu_device->super.device_taskpool_register   = parsec_cuda_taskpool_register;
        gpu_device->super.device_taskpool_unregister = parsec_cuda_taskpool_unregister;

        if (parsec_cuda_device_lookup_cudamp_floprate(major, minor, &drate, &srate, &trate, &hrate) == PARSEC_ERROR ) {
            return -1;
        }
        gpu_device->super.device_hweight = (float)streaming_multiprocessor * (float)hrate * (float)clockRate * 2e-3f;
        gpu_device->super.device_tweight = (float)streaming_multiprocessor * (float)trate * (float)clockRate * 2e-3f;
        gpu_device->super.device_sweight = (float)streaming_multiprocessor * (float)srate * (float)clockRate * 2e-3f;
        gpu_device->super.device_dweight = (float)streaming_multiprocessor * (float)drate * (float)clockRate * 2e-3f;

        if( show_caps ) {
            parsec_inform("GPU Device %d (capability %d.%d): %s\n"
                          "\tSM                 : %d\n"
                          "\tclockRate (GHz)    : %2.2f\n"
                          "\tconcurrency        : %s\n"
                          "\tcomputeMode        : %d\n"
                          "\tpeak Gflops         : double %2.4f, single %2.4f tensor %2.4f half %2.4f",
                          i, major, minor,szName,
                          streaming_multiprocessor,
                          clockRate*1e-3,
                          (concurrency == 1)? "yes": "no",
                          computemode,
                          gpu_device->super.device_dweight, gpu_device->super.device_sweight, gpu_device->super.device_tweight, gpu_device->super.device_hweight);
        }

        if( PARSEC_SUCCESS != parsec_cuda_memory_reserve(gpu_device,
                                                         cuda_memory_percentage,
                                                         cuda_memory_number_of_blocks,
                                                         cuda_memory_block_size) ) {
            free(gpu_device);
            continue;
        }

        /* Initialize internal lists */
        OBJ_CONSTRUCT(&gpu_device->gpu_mem_lru,       parsec_list_t);
        OBJ_CONSTRUCT(&gpu_device->gpu_mem_owned_lru, parsec_list_t);
        OBJ_CONSTRUCT(&gpu_device->pending,           parsec_fifo_t);

        gpu_device->sort_starting_p = NULL;
        parsec_devices_add(parsec_context, &(gpu_device->super));
    }

#if defined(PARSEC_HAVE_PEER_DEVICE_MEMORY_ACCESS)
    for( i = 0; i < ndevices; i++ ) {
        gpu_device_t *source_gpu, *target_gpu;
        int canAccessPeer;

        if( NULL == (source_gpu = (gpu_device_t*)parsec_devices_get(i)) ) continue;
        /* Skip all non CUDA devices */
        if( PARSEC_DEV_CUDA != source_gpu->super.type ) continue;

        source_gpu->peer_access_mask = 0;

        for( j = 0; j < ndevices; j++ ) {
            if( (NULL == (target_gpu = (gpu_device_t*)parsec_devices_get(j))) || (i == j) ) continue;
            /* Skip all non CUDA devices */
            if( PARSEC_DEV_CUDA != target_gpu->super.type ) continue;

            /* Communication mask */
            cudastatus = cudaDeviceCanAccessPeer( &canAccessPeer, source_gpu->cuda_index, target_gpu->cuda_index );
            PARSEC_CUDA_CHECK_ERROR( "(parsec_gpu_init) cudaDeviceCanAccessPeer ", cudastatus,
                                     {continue;} );
            if( 1 == canAccessPeer ) {
                cudastatus = cudaDeviceEnablePeerAccess( target_gpu->cuda_index, 0 );
                PARSEC_CUDA_CHECK_ERROR( "(parsec_gpu_init) cuCtxEnablePeerAccess ", cudastatus,
                                         {continue;} );
                source_gpu->peer_access_mask = (int16_t)(source_gpu->peer_access_mask | (int16_t)(1 << target_gpu->cuda_index));
            }
        }
    }
#endif

    return 0;
}

int parsec_gpu_fini(void)
{
    gpu_device_t* gpu_device;
    int i;

    for(i = 0; i < parsec_devices_enabled(); i++) {
        if( NULL == (gpu_device = (gpu_device_t*)parsec_devices_get(i)) ) continue;
        if(PARSEC_DEV_CUDA != gpu_device->super.type) continue;
        parsec_cuda_device_fini((parsec_device_t*)gpu_device);
        parsec_devices_remove((parsec_device_t*)gpu_device);
        free(gpu_device);
    }

    if( parsec_device_output != parsec_cuda_output_stream )
        parsec_output_close(parsec_cuda_output_stream);
    parsec_cuda_output_stream = -1;

    if ( cuda_lib_path ) {
        free(cuda_lib_path);
    }

    return PARSEC_SUCCESS;
}

/**
 * This function reserve the memory_percentage of the total device memory for PaRSEC.
 * This memory will be managed in chuncks of size eltsize. However, multiple chuncks
 * can be reserved in a single allocation.
 */
static int
parsec_cuda_memory_reserve( gpu_device_t* gpu_device,
                            int           memory_percentage,
                            int           number_blocks,
                            size_t        eltsize )
{
    cudaError_t status;
    (void)eltsize;

    size_t how_much_we_allocate;
    size_t total_mem, initial_free_mem;
    uint32_t mem_elem_per_gpu = 0;

    status = cudaSetDevice( gpu_device->cuda_index );
    PARSEC_CUDA_CHECK_ERROR( "(parsec_cuda_memory_reserve) cudaSetDevice ", status,
                            {continue;} );

    /* Determine how much memory we can allocate */
    cudaMemGetInfo( &initial_free_mem, &total_mem );
    if( number_blocks != -1 ) {
        if( number_blocks == 0 ) {
            parsec_warning("CUDA[%d] Invalid argument: requesting 0 bytes of memory on CUDA device %s", gpu_device->cuda_index, gpu_device->super.name);
            return PARSEC_ERROR;
        } else {
            how_much_we_allocate = number_blocks * eltsize;
        }
    } else {
        /** number_blocks == -1 means memory_percentage is used */
        how_much_we_allocate = (memory_percentage * initial_free_mem) / 100;
    }
    if( how_much_we_allocate > initial_free_mem ) {
        /** Handle the case of jokers who require more than 100% of memory,
         *  and eleventh case of computer scientists who don't know how
         *  to divide a number by another
         */
        parsec_warning("CUDA[%d] Requested %zd bytes on CUDA device %s, but only %zd bytes are available -- reducing allocation to max available",
                       gpu_device->cuda_index, how_much_we_allocate, gpu_device->super.name, initial_free_mem);
        how_much_we_allocate = initial_free_mem;
    }
    if( how_much_we_allocate < eltsize ) {
        /** Handle another kind of jokers entirely, and cases of
         *  not enough memory on the device
         */
        parsec_warning("CUDA[%d] Cannot allocate at least one element on CUDA device %s", gpu_device->cuda_index, gpu_device->super.name);
        return PARSEC_ERROR;
    }

#if defined(PARSEC_GPU_CUDA_ALLOC_PER_TILE)
    size_t free_mem = initial_free_mem;
    /*
     * We allocate a bunch of tiles that will be used
     * during the computations
     */
    while( (free_mem > eltsize )
           && ((total_mem - free_mem) < how_much_we_allocate) ) {
        parsec_gpu_data_copy_t* gpu_elem;
        void *device_ptr;

        status = (cudaError_t)cudaMalloc( &device_ptr, eltsize);
        PARSEC_CUDA_CHECK_ERROR( "(parsec_cuda_memory_reserve) cudaMemAlloc ", status,
                                ({
                                    size_t _free_mem, _total_mem;
                                    cudaMemGetInfo( &_free_mem, &_total_mem );
                                    parsec_inform("GPU[%d] Per context: free mem %zu total mem %zu (allocated tiles %u)",
                                                  gpu_device->cuda_index,_free_mem, _total_mem, mem_elem_per_gpu);
                                    break;
                                }) );
        gpu_elem = OBJ_NEW(parsec_data_copy_t);
        PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                            "GPU[%d] Allocate CUDA copy %p [ref_count %d] for data [%p]",
                             gpu_device->cuda_index,gpu_elem, gpu_elem->super.obj_reference_count, NULL);
        gpu_elem->device_private = (void*)(long)device_ptr;
        gpu_elem->device_index = gpu_device->super.device_index;
        mem_elem_per_gpu++;
        OBJ_RETAIN(gpu_elem);
        PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                            "GPU[%d] Retain and insert CUDA copy %p [ref_count %d] in LRU",
                             gpu_device->cuda_index, gpu_elem, gpu_elem->super.obj_reference_count);
        parsec_list_nolock_push_back( &gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_elem );
        cudaMemGetInfo( &free_mem, &total_mem );
    }
    if( 0 == mem_elem_per_gpu && parsec_list_nolock_is_empty( &gpu_device->gpu_mem_lru ) ) {
        parsec_warning("GPU[%d] Cannot allocate memory on GPU %s. Skip it!", gpu_device->cuda_index, gpu_device->super.name);
    }
    else {
        PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                             "GPU[%d] Allocate %u tiles on the GPU memory",
                             gpu_device->cuda_index, mem_elem_per_gpu );
    }
    PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                         "GPU[%d] Allocate %u tiles on the GPU memory", gpu_device->cuda_index, mem_elem_per_gpu);
#else
    if( NULL == gpu_device->memory ) {
        void* base_ptr;
        /* We allocate all the memory on the GPU and we use our memory management. */
        /* This computation leads to allocating more than available if we asked for more than GPU memory */
        mem_elem_per_gpu = (how_much_we_allocate + eltsize - 1 ) / eltsize;
        size_t total_size = (size_t)mem_elem_per_gpu * eltsize;
        if (total_size > initial_free_mem) {
            /* Mapping more than 100% of GPU memory is obviously wrong */
            /* Mapping exactly 100% of the GPU memory ends up producing errors about __global__ function call is not configured */
            /* Mapping 95% works with low-end GPUs like 1060, how much to let available for cuda runtime, I don't know how to calculate */
            total_size = (size_t)((int)(.95*initial_free_mem / eltsize)) * eltsize;
            mem_elem_per_gpu = total_size / eltsize;
        }
        status = (cudaError_t)cudaMalloc(&base_ptr, total_size);
        PARSEC_CUDA_CHECK_ERROR( "(parsec_cuda_memory_reserve) cudaMalloc ", status,
                                 ({ parsec_warning("GPU[%d] Allocating memory on the GPU device failed", gpu_device->cuda_index); }) );

        gpu_device->memory = zone_malloc_init( base_ptr, mem_elem_per_gpu, eltsize );

        if( gpu_device->memory == NULL ) {
            parsec_warning("GPU[%d] Cannot allocate memory on GPU %s. Skip it!", gpu_device->cuda_index, gpu_device->super.name);
            return PARSEC_ERROR;
        }
        PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                            "GPU[%d] Allocate %u segment of size %d on the GPU memory",
                             gpu_device->cuda_index, mem_elem_per_gpu, eltsize );
    }
#endif

    return PARSEC_SUCCESS;
}

static void parsec_cuda_memory_release_list(gpu_device_t* gpu_device,
                                           parsec_list_t* list)
{
    parsec_list_item_t* item;

    while(NULL != (item = parsec_list_nolock_pop_front(list)) ) {
        parsec_gpu_data_copy_t* gpu_copy = (parsec_gpu_data_copy_t*)item;
        parsec_data_t* original = gpu_copy->original;

        PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                            "Release CUDA copy %p (device_ptr %p) [ref_count %d: must be 1], attached to %p, in map %p",
                            gpu_copy, gpu_copy->device_private, gpu_copy->super.super.obj_reference_count,
                            original, (NULL != original ? original->dc : NULL));
        assert( gpu_copy->device_index == gpu_device->super.device_index );
        if( DATA_COHERENCY_OWNED == gpu_copy->coherency_state ) {
            parsec_warning("GPU[%d] still OWNS the master memory copy for data %d and it is discarding it!",
                          gpu_device->cuda_index, original->key);
        }
#if defined(PARSEC_GPU_CUDA_ALLOC_PER_TILE)
        cudaFree( gpu_copy->device_private );
#else
        zone_free( gpu_device->memory, (void*)gpu_copy->device_private );
#endif
        gpu_copy->device_private = NULL;

        /* At this point the data copies should have no attachement to a data_t. Thus,
         * before we get here (aka below parsec_fini), the destructor of the data
         * collection must have been called, releasing all the copies.
         */
        OBJ_RELEASE(gpu_copy); assert(NULL == gpu_copy);
    }
}

/**
 * This function release the CUDA memory reserved for this device.
 *
 * One has to notice that all the data available on the GPU is stored in one of
 * the two used to keep track of the allocated data, either the gpu_mem_lru or
 * the gpu_mem_owner_lru. Thus, going over all the elements in these two lists
 * should be enough to enforce a clean release.
 */
static int
parsec_cuda_memory_release( gpu_device_t* gpu_device )
{
    cudaError_t status;

    status = cudaSetDevice( gpu_device->cuda_index );
    PARSEC_CUDA_CHECK_ERROR( "(parsec_cuda_memory_release) cudaSetDevice ", status,
                            {continue;} );

    /* Free all memory on GPU */
    parsec_cuda_memory_release_list(gpu_device, &gpu_device->gpu_mem_lru);
    parsec_cuda_memory_release_list(gpu_device, &gpu_device->gpu_mem_owned_lru);

#if !defined(PARSEC_GPU_CUDA_ALLOC_PER_TILE)
    assert( NULL != gpu_device->memory );
    void* ptr = zone_malloc_fini(&gpu_device->memory);
    status = cudaFree(ptr);
    PARSEC_CUDA_CHECK_ERROR( "(parsec_cuda_memory_release) cudaFree ", status,
                             { parsec_warning("Failed to free the GPU backend memory."); } );
#endif

    return PARSEC_SUCCESS;
}

/**
 * Try to find memory space to move all data on the GPU. We attach a device_elem to
 * a memory_elem as soon as a device_elem is available. If we fail to find enough
 * available elements, we push all the elements handled during this allocation
 * back into the pool of available device_elem, to be picked up by another call
 * (this call will remove them from the current task).
 * Returns:
 *   PARSEC_HOOK_RETURN_DONE:  All gpu_mem/mem_elem have been initialized
 *   PARSEC_HOOK_RETURN_AGAIN: At least one flow is marked under transfer, task cannot be scheduled yet
 *   PARSEC_HOOK_RETURN_NEXT:  The task needs to rescheduled
 */
static inline int
parsec_gpu_data_reserve_device_space( gpu_device_t* gpu_device,
                                      parsec_gpu_task_t *gpu_task )
{
    parsec_task_t *this_task = gpu_task->ec;
    parsec_gpu_data_copy_t* temp_loc[MAX_PARAM_COUNT], *gpu_elem, *lru_gpu_elem;
    parsec_data_t* master, *oldmaster;
    const parsec_flow_t *flow;
    int i, j, data_avail_epoch = 0;
#if defined(PARSEC_DEBUG_NOISIER)
    char task_name[MAX_TASK_STRLEN];
    parsec_task_snprintf(task_name, MAX_TASK_STRLEN, this_task);
#endif  /* defined(PARSEC_DEBUG_NOISIER) */

    /**
     * Parse all the input and output flows of data and ensure all have
     * corresponding data on the GPU available.
     */
    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        flow = gpu_task->flow[i];
        assert( flow && (flow->flow_index == i) );

        /* Skip CTL flows only */
        if(!(flow->flow_flags)) continue;

        PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                             "GPU[%d]:%s: Investigating flow %s:%i",
                             gpu_device->cuda_index, task_name, flow->name, i);
        temp_loc[i] = NULL;
        master   = this_task->data[i].data_in->original;
        gpu_elem = PARSEC_DATA_GET_COPY(master, gpu_device->super.device_index);
        this_task->data[i].data_out = gpu_elem;

        /* There is already a copy on the device */
        if( NULL != gpu_elem ) {
            PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                                 "GPU[%d]:%s: Flow %s:%i has a copy on the device %p%s",
                                 gpu_device->cuda_index, task_name,
                                 flow->name, i, gpu_elem,
                                 gpu_elem->data_transfer_status == DATA_STATUS_UNDER_TRANSFER ? " [in transfer]" : "");
            if ( flow->flow_flags & FLOW_ACCESS_WRITE &&
                gpu_elem->data_transfer_status == DATA_STATUS_UNDER_TRANSFER ) {
                PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                                     "GPU[%d]:%s: Copy %p accessed RW, Descheduling while transfering data",
                                     gpu_device->cuda_index, task_name,
                                     gpu_elem);
                SET_HIGHEST_PRIORITY(gpu_task, parsec_execution_context_priority_comparator);
                return PARSEC_HOOK_RETURN_AGAIN;
            }
            continue;
        }

#if !defined(PARSEC_GPU_CUDA_ALLOC_PER_TILE)
        gpu_elem = OBJ_NEW(parsec_data_copy_t);
        PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                             "GPU[%d]:%s: Allocate CUDA copy %p [ref_count %d] for data %p",
                             gpu_device->cuda_index, task_name,
                             gpu_elem, gpu_elem->super.super.obj_reference_count, master);
      malloc_data:
        gpu_elem->device_private = zone_malloc(gpu_device->memory, master->nb_elts);
        if( NULL == gpu_elem->device_private ) {
#endif

          find_another_data:
            /* Look for a data_copy to free */
            lru_gpu_elem = (parsec_gpu_data_copy_t*)parsec_list_nolock_pop_front(&gpu_device->gpu_mem_lru);
            if( NULL == lru_gpu_elem ) {
                /* We can't find enough room on the GPU. Insert the tiles in the begining of
                 * the LRU (in order to be reused asap) and return without scheduling the task.
                 */
#if defined(PARSEC_DEBUG_NOISIER)
                PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                                     "GPU[%d]:%s:\tRequest space on GPU failed for flow %s index %d/%d for task %s",
                                     gpu_device->cuda_index, task_name,
                                     flow->name, i, this_task->task_class->nb_flows, task_name );
#endif  /* defined(PARSEC_DEBUG_NOISIER) */
                for( j = 0; j < i; j++ ) {
                    if( NULL == temp_loc[j] ) continue;
                    PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                                         "GPU[%d]:%s:\tAdd copy %p back to the LRU list",
                                         gpu_device->cuda_index, task_name,
                                         temp_loc[j]);
                    /* push them at the head to reach them again at the next iteration */
                    parsec_list_nolock_push_front(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)temp_loc[j]);
                }
#if !defined(PARSEC_GPU_CUDA_ALLOC_PER_TILE)
                OBJ_RELEASE(gpu_elem);
#endif
                return PARSEC_HOOK_RETURN_NEXT;
            }

            PARSEC_LIST_ITEM_SINGLETON(lru_gpu_elem);
            PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                                 "GPU[%d]:%s: Evaluate LRU-retrieved CUDA copy %p [ref_count %d] original %p",
                                 gpu_device->cuda_index, task_name,
                                 lru_gpu_elem, lru_gpu_elem->super.super.obj_reference_count,
                                 lru_gpu_elem->original);

            /* If there are pending readers, let the gpu_elem loose. This is a weak coordination
             * protocol between here and the parsec_gpu_data_stage_in, where the readers don't necessarily
             * always remove the data from the LRU.
             */
            if( 0 != lru_gpu_elem->readers ) {
                PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                                     "GPU[%d]:%s: Drop LRU-retrieved CUDA copy %p [readers %d] original %p",
                                     gpu_device->cuda_index, task_name,
                                     lru_gpu_elem, lru_gpu_elem->readers, lru_gpu_elem->original);
                goto find_another_data; // TODO: add an assert of some sort to check for leaks here?
            }

            /* Make sure the new GPU element is clean and ready to be used */
            assert( master != lru_gpu_elem->original );
            if ( NULL != lru_gpu_elem->original ) {
                /* Let's check we're not trying to steal one of our own data */
                oldmaster = lru_gpu_elem->original;
                for( j = 0; j < i; j++ ) {
                    if( NULL == this_task->data[j].data_in ) continue;
                    if( this_task->data[j].data_in->original == oldmaster ) {
                        PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                                             "GPU[%d]:%s: Drop LRU-retrieved CUDA copy %p [already in use by same task %d:%d] original %p",
                                             gpu_device->cuda_index, task_name,
                                             lru_gpu_elem, i, j, lru_gpu_elem->original);
                        /* If we are the owner of this tile we need to make sure it remains available for
                         * other tasks or we run in deadlock situations.
                         */
                        if( temp_loc[j] != lru_gpu_elem )
                            temp_loc[j] = lru_gpu_elem;
#if defined(PARSEC_DEBUG_NOISIER)
                        /* Make sure the data copy is indeed referenced from the current task */
                        for( j = 0; j < i; j++ ) {
                            if( lru_gpu_elem == temp_loc[j] ) break;
                        }
                        assert( j < i );
#endif  /* defined(PARSEC_DEBUG_NOISIER) */
                        goto find_another_data;
                    }
                }

                /* The data is not used, and it's not one of ours: we can free it or reuse it */
                PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                                     "GPU[%d]:%s:\ttask %s:%d repurpose copy %p to data %p instead of %p",
                                     gpu_device->cuda_index, task_name, this_task->task_class->name, i, lru_gpu_elem, master, oldmaster);
                parsec_data_copy_detach(oldmaster, lru_gpu_elem, gpu_device->super.device_index);
            }
            else {
                PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                                     "GPU[%d]:%s:\ttask %s:%d found detached memory from previously destructed data %p",
                                     gpu_device->cuda_index, task_name, this_task->task_class->name, i, lru_gpu_elem);
            }
#if !defined(PARSEC_GPU_CUDA_ALLOC_PER_TILE)
                /* Let's free this space, and try again to malloc some space */
            zone_free( gpu_device->memory, (void*)(lru_gpu_elem->device_private) );
            lru_gpu_elem->device_private = NULL;
            data_avail_epoch++;
            PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                                 "GPU[%d]:%s: Release LRU-retrieved CUDA copy %p [ref_count %d: must be 1]",
                                 gpu_device->cuda_index, task_name,
                                 lru_gpu_elem, lru_gpu_elem->super.super.obj_reference_count);
            OBJ_RELEASE(lru_gpu_elem); assert( NULL == lru_gpu_elem );
            goto malloc_data;
        }
#else
        gpu_elem = lru_gpu_elem;
#endif
        assert( 0 == gpu_elem->readers );
        gpu_elem->coherency_state = DATA_COHERENCY_INVALID;
        gpu_elem->version = 0;
        parsec_data_copy_attach(master, gpu_elem, gpu_device->super.device_index);
        this_task->data[i].data_out = gpu_elem;
        temp_loc[i] = gpu_elem;
        PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                             "GPU[%d]:%s: Retain and insert CUDA copy %p [ref_count %d] in LRU",
                             gpu_device->cuda_index, task_name,
                             gpu_elem, gpu_elem->super.super.obj_reference_count);
        parsec_list_nolock_push_back(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_elem);
    }
    if( data_avail_epoch ) {
        gpu_device->data_avail_epoch++;
    }
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * If the most current version of the data is not yet available on the GPU memory
 * schedule a transfer.
 * Returns:
 *    0: The most recent version of the data is already available on the GPU
 *    1: A copy has been scheduled on the corresponding stream
 *   -1: A copy cannot be issued due to CUDA.
 */
static inline int
parsec_gpu_data_stage_in( gpu_device_t* gpu_device,
                          int32_t type,
                          parsec_data_pair_t* task_data,
                          parsec_gpu_task_t *gpu_task,
                          parsec_gpu_exec_stream_t *gpu_stream )
{
    parsec_data_copy_t* in_elem = task_data->data_in;
    parsec_data_t* original = in_elem->original;
    parsec_gpu_data_copy_t* gpu_elem = task_data->data_out;
    int transfer_from = -1;

    /**
     * If the data will be accessed in write mode, remove it from any GPU data management
     * lists until the task is completed.
     */
    if( FLOW_ACCESS_WRITE & type ) {
        if (gpu_elem->readers > 0 ) {
            if( !((1 == gpu_elem->readers) && (FLOW_ACCESS_READ & type)) ) {
                parsec_warning("GPU[%d]:\tWrite access to data copy %p with existing readers [%d] "
                               "(possible anti-dependency,\n"
                               "or concurrent accesses), please prevent that with CTL dependencies\n",
                               gpu_device->cuda_index, gpu_elem, gpu_elem->readers);
                return -1;
            }
        }
        PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                             "GPU[%1d]:\tDetach writable CUDA copy %p from any lists",
                             gpu_device->cuda_index, gpu_elem);
        /* make sure the element is not in any tracking lists */
        parsec_list_item_ring_chop((parsec_list_item_t*)gpu_elem);
        PARSEC_LIST_ITEM_SINGLETON(gpu_elem);
        gpu_elem->version++;  /* on to the next version */
    }

    /* DtoD copy, if data is read only, then we go back to CPU copy, and fetch data from CPU (HtoD) */
    if( (in_elem != original->device_copies[0]) &&
        (in_elem->version == original->device_copies[0]->version) ) {
        /* We should never enter here, as long as we don't foward the GPU data to the input */
        assert(!((in_elem != original->device_copies[0]) &&
                 (in_elem->version == original->device_copies[0]->version)));
        parsec_data_copy_release(in_elem);  /* release the copy in GPU */
        task_data->data_in = original->device_copies[0];
        in_elem = task_data->data_in;
        OBJ_RETAIN(in_elem);  /* retain the corresponding CPU copy */
    }

    transfer_from = parsec_data_transfer_ownership_to_copy(original, gpu_device->super.device_index, (uint8_t)type);
    gpu_device->super.required_data_in += original->nb_elts;
    if( -1 != transfer_from ) {
        cudaError_t status;

        PARSEC_DEBUG_VERBOSE(10, parsec_cuda_output_stream,
                             "GPU[%d]:\t\tMove H2D data copy %p [key %x] of %d bytes\t(Host: v:%d, ptr:%p / Device: v:%d, ptr:%p)",
                             gpu_device->cuda_index, gpu_elem, original->key, original->nb_elts,
                             in_elem->version, in_elem->device_private,
                             gpu_elem->version, (void*)gpu_elem->device_private);

        assert( gpu_elem->version <= in_elem->version );
        assert((gpu_elem->version != in_elem->version) || (gpu_elem->data_transfer_status == DATA_STATUS_NOT_TRANSFER));

#if defined(PARSEC_PROF_TRACE)
        if( gpu_stream->prof_event_track_enable ) {
            parsec_task_t *this_task = gpu_task->ec;

            assert(-1 != gpu_stream->prof_event_key_start);
            PARSEC_PROFILING_TRACE(gpu_stream->profiling,
                                   gpu_stream->prof_event_key_start,
                                   this_task->task_class->key_functions->key_hash(this_task->task_class->make_key(this_task->taskpool, this_task->locals), 64, NULL),
                                   this_task->taskpool->taskpool_id,
                                   &original);
        }
#endif

        /* Push data into the GPU */
        status = (cudaError_t)cudaMemcpyAsync( gpu_elem->device_private,
                                               in_elem->device_private, original->nb_elts,
                                               cudaMemcpyHostToDevice,
                                               gpu_stream->cuda_stream );
        PARSEC_CUDA_CHECK_ERROR( "cudaMemcpyAsync to device ", status,
                                 { parsec_warning("<<%p>> -> <<%p>> [%d]", in_elem->device_private, gpu_elem->device_private, original->nb_elts);
                                     return -1; } );
        gpu_device->super.transferred_data_in += original->nb_elts;

        /* update the data version in GPU immediately, and mark the data under transfer */
        assert((gpu_elem->version != in_elem->version) || (gpu_elem->data_transfer_status == DATA_STATUS_NOT_TRANSFER));
        gpu_elem->version = in_elem->version;
        gpu_elem->data_transfer_status = DATA_STATUS_UNDER_TRANSFER;
        gpu_elem->push_task = gpu_task->ec;  /* only the task who does the transfer can modify the data status later. */
        return 1;
    }
    assert( gpu_elem->data_transfer_status == DATA_STATUS_COMPLETE_TRANSFER ||
            gpu_elem->data_transfer_status == DATA_STATUS_UNDER_TRANSFER);

    PARSEC_DEBUG_VERBOSE(10, parsec_cuda_output_stream,
                         "GPU[%d]:\t\tNO Move H2D for data copy %p [key %x] of %d bytes (host v:%d / devive v:%d)",
                         gpu_device->cuda_index, gpu_elem, original->key, original->nb_elts,
                         in_elem->version, gpu_elem->version);
    /* TODO: data keeps the same coherence flags as before */
    return 0;
}

void* parsec_gpu_pop_workspace(gpu_device_t* gpu_device, parsec_gpu_exec_stream_t* gpu_stream, size_t size)
{
    (void)gpu_device; (void)gpu_stream; (void)size;
    void *work = NULL;

#if !defined(PARSEC_GPU_CUDA_ALLOC_PER_TILE)
    if (gpu_stream->workspace == NULL) {
        gpu_stream->workspace = (parsec_gpu_workspace_t *)malloc(sizeof(parsec_gpu_workspace_t));
        gpu_stream->workspace->total_workspace = PARSEC_GPU_MAX_WORKSPACE;
        gpu_stream->workspace->stack_head = PARSEC_GPU_MAX_WORKSPACE - 1;

        for( int i = 0; i < PARSEC_GPU_MAX_WORKSPACE; i++ ) {
            gpu_stream->workspace->workspace[i] = zone_malloc( gpu_device->memory, size);
        }
    }
    assert (gpu_stream->workspace->stack_head >= 0);
    work = gpu_stream->workspace->workspace[gpu_stream->workspace->stack_head];
    gpu_stream->workspace->stack_head --;
#endif /* !defined(PARSEC_GPU_CUDA_ALLOC_PER_TILE) */
    return work;
}

int parsec_gpu_push_workspace(gpu_device_t* gpu_device, parsec_gpu_exec_stream_t* gpu_stream)
{
    (void)gpu_device; (void)gpu_stream;
#if !defined(PARSEC_GPU_CUDA_ALLOC_PER_TILE)
    gpu_stream->workspace->stack_head ++;
    assert (gpu_stream->workspace->stack_head < PARSEC_GPU_MAX_WORKSPACE);
#endif /* !defined(PARSEC_GPU_CUDA_ALLOC_PER_TILE) */
    return 0;
}

int parsec_gpu_free_workspace(gpu_device_t * gpu_device)
{
    (void)gpu_device;
#if !defined(PARSEC_GPU_CUDA_ALLOC_PER_TILE)
    int i, j;
    for( i = 0; i < gpu_device->max_exec_streams; i++ ) {
        parsec_gpu_exec_stream_t *gpu_stream = &(gpu_device->exec_stream[i]);
        if (gpu_stream->workspace != NULL) {
            for (j = 0; j < gpu_stream->workspace->total_workspace; j++) {
                zone_free( gpu_device->memory, gpu_stream->workspace->workspace[j] );
            }
            free(gpu_stream->workspace);
            gpu_stream->workspace = NULL;
        }
    }
#endif /* !defined(PARSEC_GPU_CUDA_ALLOC_PER_TILE) */
    return 0;
}

static inline int
parsec_gpu_check_space_needed(gpu_device_t *gpu_device,
                              parsec_gpu_task_t *gpu_task)
{
    int i;
    int space_needed = 0;
    parsec_task_t *this_task = gpu_task->ec;
    parsec_data_t *original;
    parsec_data_copy_t *data;
    const parsec_flow_t *flow;

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        flow = gpu_task->flow[i];
        if(!(flow->flow_flags)) continue;

        data = this_task->data[i].data_in;
        original = data->original;
        if( NULL != PARSEC_DATA_GET_COPY(original, gpu_device->super.device_index) ) {
            continue;
        }
        if(flow->flow_flags & FLOW_ACCESS_READ)
            space_needed++;
    }
    return space_needed;
}

void dump_list(parsec_list_t *list)
{
    parsec_list_item_t *p = (parsec_list_item_t *)list->ghost_element.list_next;
    while (p != &(list->ghost_element)) {
        p = (parsec_list_item_t *)p->list_next;
    }
}


int parsec_gpu_sort_pending_list(gpu_device_t *gpu_device)
{
    parsec_list_t *sort_list = gpu_device->exec_stream[0].fifo_pending;

    if (parsec_list_nolock_is_empty(sort_list) ) { /* list is empty */
        return 0;
    }

    if (gpu_device->sort_starting_p == NULL || !parsec_list_nolock_contains(sort_list, gpu_device->sort_starting_p) ) {
        gpu_device->sort_starting_p = (parsec_list_item_t*)sort_list->ghost_element.list_next;
    }

    /* p is head */
    parsec_list_item_t *p = gpu_device->sort_starting_p;
    int i, j, NB_SORT = 10, space_q, space_min;

    parsec_list_item_t *q, *prev_p, *min_p;
    for (i = 0; i < NB_SORT; i++) {
        if ( p == &(sort_list->ghost_element) ) {
            break;
        }
        min_p = p; /* assume the minimum one is the first one p */
        q = (parsec_list_item_t*)min_p->list_next;
        space_min = parsec_gpu_check_space_needed(gpu_device, (parsec_gpu_task_t*)min_p);
        for (j = i+1; j < NB_SORT; j++) {
            if ( q == &(sort_list->ghost_element) ) {
                break;
            }
            space_q = parsec_gpu_check_space_needed(gpu_device, (parsec_gpu_task_t*)q);
            if ( space_min > space_q ) {
                min_p = q;
                space_min = space_q;
            }
            q = (parsec_list_item_t*)q->list_next;

        }
        if (min_p != p) { /* minimum is not the first one, let's insert min_p before p */
            /* take min_p out */
            parsec_list_item_ring_chop(min_p);
            PARSEC_LIST_ITEM_SINGLETON(min_p);
            prev_p = (parsec_list_item_t*)p->list_prev;

            /* insert min_p after prev_p */
            parsec_list_nolock_add_after( sort_list, prev_p, min_p);
        }
        p = (parsec_list_item_t*)min_p->list_next;
    }

    return 0;
}


/**
 * Try to find the best device to execute the kernel based on the compute
 * capability of the card.
 *
 * Returns:
 *  > 1    - if the kernel should be executed by the a GPU
 *  0 or 1 - if the kernel should be executed by some other meaning (in this case the
 *         execution context is not released).
 * -1      - if the kernel is scheduled to be executed on a GPU.
 */

/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for transfers from the GPU into
 * the main memory. The synchronization on each stream is based on CUDA events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
int parsec_gpu_get_best_device( parsec_task_t* this_task, double ratio )
{
    int i, dev_index = -1, data_index = 0;
    parsec_taskpool_t* tp = this_task->taskpool;

    /* Step one: Find the first data in WRITE mode stored on a GPU */
    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        if( (NULL != this_task->task_class->out[i]) &&
            (this_task->task_class->out[i]->flow_flags & FLOW_ACCESS_WRITE) ) {
            data_index = this_task->task_class->out[i]->flow_index;
            dev_index  = this_task->data[data_index].data_in->original->owner_device;
            if (dev_index > 1) {
                break;
            }
        }
    }
    assert(dev_index >= 0);

    /* 0 is CPU, and 1 is recursive device */
    if( dev_index <= 1 ) {  /* This is the first time we see this data for a GPU.
                             * Let's decide which GPU will work on it. */
        int best_index = 0;  /* default value: first CPU device */
        float weight, best_weight = parsec_device_load[0] + ratio * parsec_device_sweight[0];

        /* Start at 2, to skip the recursive body */
        for( dev_index = 2; dev_index < parsec_devices_enabled(); dev_index++ ) {
            /* Skip the device if it is not configured */
            if(!(tp->devices_index_mask & (1 << dev_index))) continue;
            weight = parsec_device_load[dev_index] + ratio * parsec_device_sweight[dev_index];
            if( best_weight > weight ) {
                best_index = dev_index;
                best_weight = weight;
            }
        }
        assert( best_index != 1 );
        dev_index = best_index;
    }

    return dev_index;
}

#if PARSEC_GPU_USE_PRIORITIES

static inline parsec_list_item_t* parsec_push_task_ordered( parsec_list_t* list,
                                                          parsec_list_item_t* elem )
{
    parsec_list_nolock_push_sorted(list, elem, parsec_execution_context_priority_comparator);
    return elem;
}
#define PARSEC_PUSH_TASK parsec_push_task_ordered
#else
#define PARSEC_PUSH_TASK parsec_list_nolock_push_back
#endif

static int
parsec_gpu_callback_complete_push(gpu_device_t              *gpu_device,
                                  parsec_gpu_task_t        **gpu_task,
                                  parsec_gpu_exec_stream_t  *gpu_stream)
{
    (void)gpu_device;
    (void)gpu_stream;

    parsec_gpu_task_t *gtask = *gpu_task;
    parsec_task_t *task;
    int32_t i;
#if defined(PARSEC_DEBUG_NOISIER)
    char task_str[MAX_TASK_STRLEN];
#endif
    const parsec_flow_t        *flow;
    /**
     * Even though cuda event return success, the PUSH may not be
     * completed if no PUSH is required by this task and the PUSH is
     * actually done by another task, so we need to check if the data is
     * actually ready to use
     */
    assert(gpu_stream == &(gpu_device->exec_stream[0]));
    task = gtask->ec;
#if defined(PARSEC_DEBUG_NOISIER)
    PARSEC_DEBUG_VERBOSE(19, parsec_cuda_output_stream,
                         "GPU[%d]: parsec_gpu_callback_complete_push, PUSH of %s",
                         gpu_device->cuda_index, parsec_task_snprintf(task_str, MAX_TASK_STRLEN, task));
#endif
    for( i = 0; i < task->task_class->nb_flows; i++ ) {
        flow = gtask->flow[i];
        assert( flow );
        assert( flow->flow_index == i );
        if(!flow->flow_flags) continue;
        if(task->data[i].data_out->push_task == task ) {   /* only the task who did this PUSH can modify the status */
            task->data[i].data_out->data_transfer_status = DATA_STATUS_COMPLETE_TRANSFER;
            task->data[i].data_out->push_task = NULL;
            continue;
        }
        assert(task->data[i].data_out->data_transfer_status == DATA_STATUS_COMPLETE_TRANSFER);
        if( task->data[i].data_out->data_transfer_status != DATA_STATUS_COMPLETE_TRANSFER ) {  /* data is not ready */
            /**
             * As long as we have only one stream to push the data on the GPU we should never
             * end up in this case. Remove previous assert if changed.
             */
            return -1;
        }
    }
    gtask->complete_stage = NULL;
    return 0;
}

static inline int
progress_stream( gpu_device_t* gpu_device,
                 parsec_gpu_exec_stream_t* stream,
                 advance_task_function_t upstream_progress_fct,
                 parsec_gpu_task_t* task,
                 parsec_gpu_task_t** out_task )
{
    advance_task_function_t progress_fct;
    int saved_rc = 0, rc;

    /* We always handle the tasks in order. Thus if we got a new task, add it to the
     * local list (possibly by reordering the list). Also, as we can return a single
     * task first try to see if anything completed. */
    if( NULL != task ) {
        PARSEC_PUSH_TASK(stream->fifo_pending, (parsec_list_item_t*)task);
        task = NULL;
    }
    *out_task = NULL;
    progress_fct = upstream_progress_fct;

    if( NULL != stream->tasks[stream->end] ) {
        rc = cudaEventQuery(stream->events[stream->end]);
        if( cudaSuccess == rc ) {
            /* Save the task for the next step */
            task = *out_task = stream->tasks[stream->end];
            PARSEC_DEBUG_VERBOSE(19, parsec_cuda_output_stream,
                                 "GPU[%d]: Completed %s(task %p) priority %d on stream %s{%p}",
                                 gpu_device->cuda_index,
                                 task->ec->task_class->name, (void*)task->ec, task->ec->priority,
                                 stream->name, (void*)stream);
            stream->tasks[stream->end]    = NULL;
            stream->end = (stream->end + 1) % stream->max_events;

#if defined(PARSEC_PROF_TRACE)
            if( stream->prof_event_track_enable ) {
                PARSEC_TASK_PROF_TRACE(stream->profiling,
                                       (-1 == stream->prof_event_key_end ?
                                        PARSEC_PROF_FUNC_KEY_END(task->ec->taskpool,
                                                                 task->ec->task_class->task_class_id) :
                                        stream->prof_event_key_end),
                                       task->ec);
            }
#endif /* (PARSEC_PROF_TRACE) */

            rc = 0;
            if (task->complete_stage)
                rc = task->complete_stage(gpu_device, out_task, stream);
            /* the task can be withdrawn by the system */
            return rc;
        }
        if( cudaErrorNotReady != rc ) {
            PARSEC_CUDA_CHECK_ERROR( "(progress_stream) cudaEventQuery ", rc,
                                     {return -1;} );
        }
    }

 grab_a_task:
    if( NULL == stream->tasks[stream->start] ) {  /* there is room on the stream */
        task = (parsec_gpu_task_t*)parsec_list_nolock_pop_front(stream->fifo_pending);  /* get the best task */
    }
    if( NULL == task ) {  /* No tasks, we're done */
        return saved_rc;
    }
    PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t*)task);

    assert( NULL == stream->tasks[stream->start] );
    /**
     * In case the task is succesfully progressed, the corresponding profiling
     * event is triggered.
     */
    if ( NULL == upstream_progress_fct ) {
        /* Grab the submit function */
        progress_fct = task->submit;
#if defined(PARSEC_DEBUG_PARANOID)
        int i;
        const parsec_flow_t *flow;
        for( i = 0; i < task->ec->task_class->nb_flows; i++ ) {
            flow = task->flow[i];
            if(!flow->flow_flags) continue;
            assert(task->ec->data[i].data_out->data_transfer_status == DATA_STATUS_COMPLETE_TRANSFER);
        }
#endif /* defined(PARSEC_DEBUG_PARANOID) */
    }
    rc = progress_fct( gpu_device, task, stream );
    if( 0 > rc ) {
        if( PARSEC_HOOK_RETURN_AGAIN != rc) {
            *out_task = task;
            return rc;
        }

        PARSEC_PUSH_TASK(stream->fifo_pending, (parsec_list_item_t*)task);
        PARSEC_DEBUG_VERBOSE(10, parsec_cuda_output_stream,
                             "GPU[%d]: Reschedule %s(task %p) priority %d: no room available on the GPU for data",
                             gpu_device->cuda_index, task->ec->task_class->name, (void*)task->ec, task->ec->priority);
        *out_task = NULL;
        return 0;
    } else {
        /**
         * Do not skip the cuda event generation. The problem is that some of the inputs
         * might be in the pipe of being transferred to the GPU. If we activate this task
         * too early, it might get executed before the data is available on the GPU.
         * Obviously, this lead to incorrect results.
         */
        rc = cudaEventRecord( stream->events[stream->start], stream->cuda_stream );
        assert(cudaSuccess == rc);
        stream->tasks[stream->start]    = task;
        stream->start = (stream->start + 1) % stream->max_events;
        PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                             "GPU[%d]: Submitted %s(task %p) priority %d on stream %s{%p}",
                             gpu_device->cuda_index,
                             task->ec->task_class->name, (void*)task->ec, task->ec->priority,
                             stream->name, (void*)stream);
    }
    task = NULL;
    goto grab_a_task;
}

void dump_exec_stream(parsec_gpu_exec_stream_t* exec_stream)
{
    char task_str[128];
    int i;

    parsec_debug_verbose(0, parsec_cuda_output_stream,
                         "Dev: CUDA stream %d{%p} [events = %d, start = %d, end = %d, executed = %d]",
                         exec_stream->name, exec_stream, exec_stream->max_events, exec_stream->start, exec_stream->end,
                         exec_stream->executed);
    for( i = 0; i < exec_stream->max_events; i++ ) {
        if( NULL == exec_stream->tasks[i] ) continue;
        parsec_debug_verbose(0, parsec_cuda_output_stream,
                             "    %d: %s", i, parsec_task_snprintf(task_str, 128, exec_stream->tasks[i]->ec));
    }
    /* Don't yet dump the fifo_pending queue */
}

void dump_GPU_state(gpu_device_t* gpu_device)
{
    int i;

    parsec_output(parsec_cuda_output_stream, "\n\n");
    parsec_output(parsec_cuda_output_stream, "Device %d:%d (%p) epoch\n", gpu_device->cuda_index,
                  gpu_device->super.device_index, gpu_device, gpu_device->data_avail_epoch);
    parsec_output(parsec_cuda_output_stream, "\tpeer mask %x executed tasks %llu max streams %d\n",
                  gpu_device->peer_access_mask, (unsigned long long)gpu_device->super.executed_tasks, gpu_device->max_exec_streams);
    parsec_output(parsec_cuda_output_stream, "\tstats transferred [in %llu out %llu] required [in %llu out %llu]\n",
                  (unsigned long long)gpu_device->super.transferred_data_in, (unsigned long long)gpu_device->super.transferred_data_out,
                  (unsigned long long)gpu_device->super.required_data_in, (unsigned long long)gpu_device->super.required_data_out);
    for( i = 0; i < gpu_device->max_exec_streams; i++ ) {
        dump_exec_stream(&gpu_device->exec_stream[i]);
    }
    if( !parsec_list_nolock_is_empty(&gpu_device->gpu_mem_lru) ) {
        parsec_output(parsec_cuda_output_stream, "#\n# LRU list\n#\n");
        i = 0;
        PARSEC_LIST_NOLOCK_ITERATOR(&gpu_device->gpu_mem_lru, item,
                                    {
                                        parsec_gpu_data_copy_t* gpu_copy = (parsec_gpu_data_copy_t*)item;
                                        parsec_output(parsec_cuda_output_stream, "  %d. elem %p GPU mem %p\n", i, gpu_copy, gpu_copy->device_private);
                                        parsec_dump_data_copy(gpu_copy);
                                        i++;
                                    });
    }
    if( !parsec_list_nolock_is_empty(&gpu_device->gpu_mem_owned_lru) ) {
        parsec_output(parsec_cuda_output_stream, "#\n# Owned LRU list\n#\n");
        i = 0;
        PARSEC_LIST_NOLOCK_ITERATOR(&gpu_device->gpu_mem_owned_lru, item,
                                    {
                                        parsec_gpu_data_copy_t* gpu_copy = (parsec_gpu_data_copy_t*)item;
                                        parsec_output(parsec_cuda_output_stream, "  %d. elem %p GPU mem %p\n", i, gpu_copy, gpu_copy->device_private);
                                        parsec_dump_data_copy(gpu_copy);
                                        i++;
                                    });
    }
    parsec_output(parsec_cuda_output_stream, "\n\n");
}

/**
 *  This function schedule the move of all the data required for a
 *  specific task from the main memory into the GPU memory.
 *
 *  Returns:
 *     a positive number: the number of data to be moved.
 *     -1: data cannot be moved into the GPU.
 *     -2: No more room on the GPU to move this data.
 */
int
parsec_gpu_kernel_push( gpu_device_t                    *gpu_device,
                        parsec_gpu_task_t               *gpu_task,
                        parsec_gpu_exec_stream_t        *gpu_stream)
{
    parsec_task_t *this_task = gpu_task->ec;
    const parsec_flow_t *flow;
    int i, ret = 0;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif

#if 0
    if( gpu_task->last_data_check_epoch == gpu_device->data_avail_epoch )
        return PARSEC_HOOK_RETURN_AGAIN;
#endif
    PARSEC_DEBUG_VERBOSE(10, parsec_cuda_output_stream,
                         "GPU[%d]: Try to Push %s",
                         gpu_device->cuda_index,
                         parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );

    /* Do we have enough available memory on the GPU to hold the input and output data ? */
    ret = parsec_gpu_data_reserve_device_space( gpu_device, gpu_task );
    if( ret < 0 ) {
        gpu_task->last_data_check_epoch = gpu_device->data_avail_epoch;
        return ret;
    }

    /* Initiate all necessary data transfers */
    PARSEC_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                              gpu_stream->profiling,
                              (-1 == gpu_stream->prof_event_key_start ?
                               PARSEC_PROF_FUNC_KEY_START(this_task->taskpool,
                                                          this_task->task_class->task_class_id) :
                               gpu_stream->prof_event_key_start),
                              this_task);

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        flow = gpu_task->flow[i];
        /* Skip CTL flows */
        if(!(flow->flow_flags)) continue;

        assert( NULL != parsec_data_copy_get_ptr(this_task->data[i].data_in) );

        PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                             "GPU[%1d]:\t\tIN  Data of %s <%x> on GPU",
                             gpu_device->cuda_index, flow->name,
                             this_task->data[i].data_out->original->key);
        ret = parsec_gpu_data_stage_in( gpu_device, flow->flow_flags,
                                        &(this_task->data[i]), gpu_task, gpu_stream );
        if( ret < 0 ) {
            return ret;
        }
    }

    PARSEC_DEBUG_VERBOSE(10, parsec_cuda_output_stream,
                         "GPU[%d]: Push task %s DONE",
                         gpu_device->cuda_index,
                         parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );
    gpu_task->complete_stage = parsec_gpu_callback_complete_push;
    return ret;
}

/**
 *  This function schedule the move of all the modified data for a
 *  specific task from the GPU memory into the main memory.
 *
 *  Returns: negative number if any error occured.
 *           positive: the number of data to be moved.
 */
int
parsec_gpu_kernel_pop( gpu_device_t            *gpu_device,
                      parsec_gpu_task_t        *gpu_task,
                      parsec_gpu_exec_stream_t *gpu_stream)
{
    parsec_task_t *this_task = gpu_task->ec;
    parsec_gpu_data_copy_t     *gpu_copy;
    parsec_data_t              *original;
    const parsec_flow_t        *flow;
    int return_code = 0, how_many = 0, i, update_data_epoch = 0;
    cudaError_t status;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif

    if (gpu_task->task_type == GPU_TASK_TYPE_D2HTRANSFER) {
        for( i = 0; i < this_task->locals[0].value; i++ ) {
            gpu_copy = this_task->data[i].data_out;
            original = gpu_copy->original;
            status = cudaMemcpyAsync( original->device_copies[0]->device_private,
                                      gpu_copy->device_private,
                                      original->nb_elts,
                                      cudaMemcpyDeviceToHost,
                                      gpu_stream->cuda_stream );
            PARSEC_CUDA_CHECK_ERROR( "cudaMemcpyAsync from device ", status,
                                    { parsec_warning("data %s <<%p>> -> <<%p>>\n", this_task->task_class->out[i]->name,
                                                    gpu_copy->device_private, original->device_copies[0]->device_private);
                                        return_code = -2;
                                        goto release_and_return_error;} );
        }
        return return_code;
    }

    PARSEC_DEBUG_VERBOSE(10, parsec_cuda_output_stream,
                        "GPU[%d]: Try to Pop %s",
                        gpu_device->cuda_index,
                        parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        /* We need to manage all data that has been used as input, even if they were read only */

        flow = gpu_task->flow[i];
        if( 0 == flow->flow_flags )  continue;  /* control flow */

        gpu_copy = this_task->data[i].data_out;
        original = gpu_copy->original;
        assert(original == this_task->data[i].data_in->original);

        if( !(flow->flow_flags & FLOW_ACCESS_WRITE) ) {
            /* Do not propagate GPU copies to successors (temporary solution) */
            this_task->data[i].data_out = original->device_copies[0];
        }

        if( flow->flow_flags & FLOW_ACCESS_READ ) {
            gpu_copy->readers--;
            assert(gpu_copy->readers >= 0);
            if( (0 == gpu_copy->readers) &&
                !(flow->flow_flags & FLOW_ACCESS_WRITE) ) {
                PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                                     "GPU[%1d]:\tMake read-only copy %p available on flow %s",
                                     gpu_device->cuda_index, gpu_copy, flow->name);
                parsec_list_item_ring_chop((parsec_list_item_t*)gpu_copy);
                PARSEC_LIST_ITEM_SINGLETON(gpu_copy); /* TODO: singleton instead? */
                parsec_list_nolock_push_back(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_copy);
                update_data_epoch = 1;
                continue;  /* done with this element, go for the next one */
            }
            PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                                 "GPU[%1d]:\tread copy %p on flow %s has readers (%i)",
                                 gpu_device->cuda_index, gpu_copy, flow->name, gpu_copy->readers);
        }
        if( flow->flow_flags & FLOW_ACCESS_WRITE ) {
            assert( gpu_copy == parsec_data_get_copy(gpu_copy->original, gpu_device->super.device_index) );

            PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                                "GPU[%1d]:\tOUT Data copy %p for flow %s",
                                 gpu_device->cuda_index, gpu_copy, flow->name);

            /* Stage the transfer of the data back to main memory */
            gpu_device->super.required_data_out += original->nb_elts;
            assert( ((parsec_list_item_t*)gpu_copy)->list_next == (parsec_list_item_t*)gpu_copy );
            assert( ((parsec_list_item_t*)gpu_copy)->list_prev == (parsec_list_item_t*)gpu_copy );

            assert( DATA_COHERENCY_OWNED == gpu_copy->coherency_state );
            if( gpu_task->pushout & (1 << i) ) {
                /* TODO: make sure no readers are working on the CPU version */
                original = gpu_copy->original;
                PARSEC_DEBUG_VERBOSE(10, parsec_cuda_output_stream,
                                    "GPU[%d]:\tMove D2H data <%s:%x> D:%p -> H:%p requested",
                                    gpu_device->cuda_index, flow->name, original->key,
                                    (void*)gpu_copy->device_private, original->device_copies[0]->device_private);
                PARSEC_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                                         gpu_stream->profiling,
                                         (-1 == gpu_stream->prof_event_key_start ?
                                          PARSEC_PROF_FUNC_KEY_START(this_task->taskpool,
                                                                    this_task->task_class->task_class_id) :
                                          gpu_stream->prof_event_key_start),
                                         this_task);
                /* Move the data back into main memory */
                status = cudaMemcpyAsync( original->device_copies[0]->device_private,
                                          gpu_copy->device_private,
                                          original->nb_elts,
                                          cudaMemcpyDeviceToHost,
                                          gpu_stream->cuda_stream );
                PARSEC_CUDA_CHECK_ERROR( "cudaMemcpyAsync from device ", status,
                                        { parsec_warning("data %s <<%p>> -> <<%p>>\n", this_task->task_class->out[i]->name,
                                                        gpu_copy->device_private, original->device_copies[0]->device_private);
                                            return_code = -2;
                                            goto release_and_return_error;} );
                gpu_device->super.transferred_data_out += original->nb_elts; /* TODO: not hardcoded, use datatype size */
                how_many++;
            } else {
                assert( 0 == gpu_copy->readers );
            }
        }
    }

  release_and_return_error:
    if( update_data_epoch ) {
        gpu_device->data_avail_epoch++;
    }
    PARSEC_DEBUG_VERBOSE(10, parsec_cuda_output_stream,
                         "GPU[%d]: Pop %s DONE (return %d data epoch %"PRIu64")",
                         gpu_device->cuda_index,
                         parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task), return_code, gpu_device->data_avail_epoch );

    return (return_code < 0 ? return_code : how_many);
}

/**
 * Make sure all data on the device is correctly put back into the queues.
 */
int
parsec_gpu_kernel_epilog( gpu_device_t      *gpu_device,
                          parsec_gpu_task_t *gpu_task )
{
    parsec_task_t *this_task = gpu_task->ec;
    parsec_gpu_data_copy_t     *gpu_copy, *cpu_copy;
    parsec_data_t              *original;
    int i;

#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    PARSEC_DEBUG_VERBOSE(10, parsec_cuda_output_stream,
                         "GPU[%d]: Epilog of %s",
                         gpu_device->cuda_index,
                         parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );
#endif

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        /* Don't bother if there is no real data (aka. CTL or no output) */
        if(NULL == this_task->data[i].data_out) continue;

        if( !(gpu_task->flow[i]->flow_flags & FLOW_ACCESS_WRITE) ) {
            /* Warning data_out for read only flows has been overwritten in pop */
            continue;
        }

        gpu_copy = this_task->data[i].data_out;
        original = gpu_copy->original;
        cpu_copy = original->device_copies[0];

        /**
         * There might be a race condition here. We can't assume the first CPU
         * version is the corresponding CPU copy, as a new CPU-bound data
         * might have been created meanwhile.
         *
         * WARNING: For now we always forward the cpu_copy to the next task, to
         * do that, we lie to the engine by updating the CPU copy to the same
         * status than the GPU copy without updating the data itself. Thus, the
         * cpu copy is really invalid. this is related to Issue #88, and the
         * fact that:
         *      - we don't forward the gpu copy as output
         *      - we always take a cpu copy as input, so it has to be in the
         *        same state as the GPU to prevent an extra data movement.
         */
        assert( DATA_COHERENCY_OWNED == gpu_copy->coherency_state );
        gpu_copy->coherency_state = DATA_COHERENCY_SHARED;
        cpu_copy->coherency_state = DATA_COHERENCY_SHARED;

        /**
         *  The cpu_copy will be updated in the completion, and at that moment
         *  the two versions will be identical.
         */
        cpu_copy->version = gpu_copy->version;

        /**
         * Let's lie to the engine by reporting that working version of this
         * data (aka. the one that GEMM worked on) is now on the CPU.
         */
        this_task->data[i].data_out = cpu_copy;

        assert( 0 == gpu_copy->readers );

        if( gpu_task->pushout & (1 << i) ) {
            PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                                 "CUDA copy %p [ref_count %d] moved to the read LRU in %s",
                                 gpu_copy, gpu_copy->super.super.obj_reference_count, __func__);
            parsec_list_item_ring_chop((parsec_list_item_t*)gpu_copy);
            PARSEC_LIST_ITEM_SINGLETON(gpu_copy);
            parsec_list_nolock_push_back(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_copy);
        } else {
            PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                                 "CUDA copy %p [ref_count %d] moved to the owned LRU in %s",
                                 gpu_copy, gpu_copy->super.super.obj_reference_count, __func__);
            parsec_list_nolock_push_back(&gpu_device->gpu_mem_owned_lru, (parsec_list_item_t*)gpu_copy);
        }
    }
    return 0;
}

/** @brief Release the CUDA copies of the data used in WRITE mode.
 *
 * @details This function can be used when the CUDA task didn't run
 *          to completion on the device (either due to an error, or
 *          simply because the body requested a reexecution on a
 *          different location). It releases the CUDA copies of the
 *          output data, allowing them to be reused by the runtime.
 *          This function has the drawback of kicking in too late,
 *          after all data transfers have been completed toward the
 *          device.
 *
 * @param [IN] gpu_device, the GPU device the the task has been
 *             supposed to execute.
 * @param [IN] gpu_task, the task that has been cancelled, and which
 *             needs it's data returned to the runtime.
 * @return Currently only success.
 */
int
parsec_gpu_kernel_cleanout( gpu_device_t      *gpu_device,
                            parsec_gpu_task_t *gpu_task )
{
    parsec_task_t *this_task = gpu_task->ec;
    parsec_gpu_data_copy_t     *gpu_copy, *cpu_copy;
    parsec_data_t              *original;
    int i, data_avail_epoch = 0;

#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    PARSEC_DEBUG_VERBOSE(10, parsec_cuda_output_stream,
                        "GPU[%d]: Cleanup of %s",
                        gpu_device->cuda_index,
                        parsec_task_snprintf(tmp, MAX_TASK_STRLEN, this_task) );
#endif

    for( i = 0; i < this_task->task_class->nb_flows; i++ ) {
        /* Don't bother if there is no real data (aka. CTL or no output) */
        if(NULL == this_task->data[i].data_out) continue;
        if( !(gpu_task->flow[i]->flow_flags & FLOW_ACCESS_WRITE) ) {
            /* Warning data_out for read only flows has been overwritten in pop */
            continue;
        }

        gpu_copy = this_task->data[i].data_out;
        original = gpu_copy->original;
        assert(gpu_copy->super.super.obj_reference_count > 1);
        /* Issue #134 */
        parsec_data_copy_detach(original, gpu_copy, gpu_device->super.device_index);
        gpu_copy->coherency_state = DATA_COHERENCY_SHARED;
        cpu_copy = original->device_copies[0];

        /**
         * Let's lie to the engine by reporting that working version of this
         * data (aka. the one that GEMM worked on) is now on the CPU.
         */
        this_task->data[i].data_out = cpu_copy;
        parsec_list_nolock_push_back(&gpu_device->gpu_mem_lru, (parsec_list_item_t*)gpu_copy);
        data_avail_epoch++;
        PARSEC_DEBUG_VERBOSE(20, parsec_cuda_output_stream,
                             "CUDA copy %p [ref_count %d] moved to the read LRU in %s\n",
                             gpu_copy, gpu_copy->super.super.obj_reference_count, __func__);
    }
    if( data_avail_epoch )  /* Update data availability epoch */
        gpu_device->data_avail_epoch++;
    return 0;
}

/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for transfers from the GPU into
 * the main memory. The synchronization on each stream is based on CUDA events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
parsec_hook_return_t
parsec_gpu_kernel_scheduler( parsec_execution_stream_t *es,
                             parsec_gpu_task_t         *gpu_task,
                             int which_gpu )
{
    gpu_device_t* gpu_device;
    cudaError_t status;
    int rc, exec_stream = 0;
    parsec_gpu_task_t *progress_task, *out_task_submit = NULL, *out_task_pop = NULL;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif

    gpu_device = (gpu_device_t*)parsec_devices_get(which_gpu);

#if defined(PARSEC_PROF_TRACE)
    PARSEC_PROFILING_TRACE_FLAGS( es->es_profile,
                                  PARSEC_PROF_FUNC_KEY_END(gpu_task->ec->taskpool,
                                                           gpu_task->ec->task_class->task_class_id),
                                  gpu_task->ec->task_class->key_functions->key_hash(gpu_task->ec->task_class->make_key(gpu_task->ec->taskpool, gpu_task->ec->locals), 64, NULL),
                                  gpu_task->ec->taskpool->taskpool_id, NULL,
                                  PARSEC_PROFILING_EVENT_RESCHEDULED );
#endif /* defined(PARSEC_PROF_TRACE) */

    /* Check the GPU status */
    rc = parsec_atomic_fetch_inc_int32( &gpu_device->mutex );
    if( 0 != rc ) {  /* I'm not the only one messing with this GPU */
        parsec_fifo_push( &(gpu_device->pending), (parsec_list_item_t*)gpu_task );
        return PARSEC_HOOK_RETURN_ASYNC;
    }

#if defined(PARSEC_PROF_TRACE)
    if( parsec_cuda_trackable_events & PARSEC_PROFILE_CUDA_TRACK_OWN )
        PARSEC_PROFILING_TRACE( es->es_profile, parsec_cuda_own_GPU_key_start,
                                (unsigned long)es, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(PARSEC_PROF_TRACE) */

    status = cudaSetDevice( gpu_device->cuda_index );
    PARSEC_CUDA_CHECK_ERROR( "(parsec_gpu_kernel_scheduler) cudaSetDevice ", status,
                             {return PARSEC_HOOK_RETURN_DISABLE;} );

 check_in_deps:
    if( NULL != gpu_task ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_cuda_output_stream,
                             "GPU[%1d]:\tUpload data (if any) for %s priority %d",
                             gpu_device->cuda_index,
                             parsec_task_snprintf(tmp, MAX_TASK_STRLEN, gpu_task->ec),
                             gpu_task->ec->priority );
    }
    rc = progress_stream( gpu_device,
                          &(gpu_device->exec_stream[0]),
                          parsec_gpu_kernel_push,
                          gpu_task, &progress_task );
    if( rc < 0 ) {  /* In case of error progress_task is the task that raised it */
        if( -1 == rc )
            goto disable_gpu;
        /* We are in the early stages, and if there no room on the GPU for a task we need to
         * delay all retries for the same task for a little while. Meanwhile, put the task back
         * trigger a device flush, and keep executing tasks that have their data on the device.
         */
        if( NULL != progress_task ) {
            PARSEC_PUSH_TASK(gpu_device->exec_stream[0].fifo_pending, (parsec_list_item_t*)progress_task);
            progress_task = NULL;
        }
        /* If we can extract data go for it, otherwise try to drain the pending tasks */
        gpu_task = parsec_gpu_create_W2R_task(gpu_device, es);
        if( NULL != gpu_task )
            goto get_data_out_of_device;
    }
    gpu_task = progress_task;

    /* Stage-in completed for this task: it is ready to be executed */
    exec_stream = (exec_stream + 1) % (gpu_device->max_exec_streams - 2);  /* Choose an exec_stream */
    if( NULL != gpu_task ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_cuda_output_stream,  "GPU[%1d]:\tExecute %s priority %d", gpu_device->cuda_index,
                             parsec_task_snprintf(tmp, MAX_TASK_STRLEN, gpu_task->ec),
                             gpu_task->ec->priority );
    }
    rc = progress_stream( gpu_device,
                          &(gpu_device->exec_stream[2+exec_stream]),
                          NULL,
                          gpu_task, &progress_task );
    if( rc < 0 ) {
        if( PARSEC_HOOK_RETURN_DISABLE == rc )
            goto disable_gpu;
        if( PARSEC_HOOK_RETURN_ASYNC != rc ) {
            /* Reschedule the task. As the chore_id has been modified,
               another incarnation of the task will be executed. */
            if( NULL != progress_task ) {
                parsec_gpu_kernel_cleanout(gpu_device, progress_task);
                __parsec_reschedule(es, progress_task->ec);
                gpu_task = progress_task;
                progress_task = NULL;
                goto remove_gpu_task;
            }
            gpu_task = NULL;
            goto fetch_task_from_shared_queue;
        }
        progress_task = NULL;
    }
    gpu_task = progress_task;
    out_task_submit = progress_task;

 get_data_out_of_device:
    if( NULL != gpu_task ) {  /* This task has completed its execution */
        PARSEC_DEBUG_VERBOSE(10, parsec_cuda_output_stream,  "GPU[%1d]:\tRetrieve data (if any) for %s priority %d", gpu_device->cuda_index,
                            parsec_task_snprintf(tmp, MAX_TASK_STRLEN, gpu_task->ec),
                            gpu_task->ec->priority );
    }
    /* Task is ready to move the data back to main memory */
    rc = progress_stream( gpu_device,
                          &(gpu_device->exec_stream[1]),
                          parsec_gpu_kernel_pop,
                          gpu_task, &progress_task );
    if( rc < 0 ) {
        if( -1 == rc )
            goto disable_gpu;
    }
    if( NULL != progress_task ) {
        /* We have a succesfully completed task. However, it is not gpu_task, as
         * it was just submitted into the data retrieval system. Instead, the task
         * ready to move into the next level is the progress_task.
         */
        gpu_task = progress_task;
        progress_task = NULL;
        goto complete_task;
    }
    gpu_task = progress_task;
    out_task_pop = progress_task;

 fetch_task_from_shared_queue:
    assert( NULL == gpu_task );
    if (0 == parsec_CUDA_sort_pending_list && out_task_submit == NULL && out_task_pop == NULL) {
        parsec_gpu_sort_pending_list(gpu_device);
    }
    gpu_task = (parsec_gpu_task_t*)parsec_fifo_try_pop( &(gpu_device->pending) );
    if( NULL != gpu_task ) {
        gpu_task->last_data_check_epoch = gpu_device->data_avail_epoch - 1;  /* force at least one tour */
        PARSEC_DEBUG_VERBOSE(10, parsec_cuda_output_stream,  "GPU[%1d]:\tGet from shared queue %s priority %d", gpu_device->cuda_index,
                             parsec_task_snprintf(tmp, MAX_TASK_STRLEN, gpu_task->ec),
                             gpu_task->ec->priority );
    }
    goto check_in_deps;

 complete_task:
    assert( NULL != gpu_task );
    PARSEC_DEBUG_VERBOSE(10, parsec_cuda_output_stream,  "GPU[%1d]:\tComplete %s",
                         gpu_device->cuda_index,
                         parsec_task_snprintf(tmp, MAX_TASK_STRLEN, gpu_task->ec));
    /* Everything went fine so far, the result is correct and back in the main memory */
    PARSEC_LIST_ITEM_SINGLETON(gpu_task);
    if (gpu_task->task_type == GPU_TASK_TYPE_D2HTRANSFER) {
        parsec_gpu_W2R_task_fini(gpu_device, gpu_task, es);
        gpu_task = progress_task;
        goto fetch_task_from_shared_queue;
    }
    parsec_gpu_kernel_epilog( gpu_device, gpu_task );
    __parsec_complete_execution( es, gpu_task->ec );
    gpu_device->super.executed_tasks++;
 remove_gpu_task:
    parsec_device_load[gpu_device->super.device_index] -= gpu_task->load;
    free( gpu_task );
    rc = parsec_atomic_fetch_dec_int32( &(gpu_device->mutex) );
    if( 1 == rc ) {  /* I was the last one */
#if defined(PARSEC_PROF_TRACE)
        if( parsec_cuda_trackable_events & PARSEC_PROFILE_CUDA_TRACK_OWN )
            PARSEC_PROFILING_TRACE( es->es_profile, parsec_cuda_own_GPU_key_end,
                                    (unsigned long)es, PROFILE_OBJECT_ID_NULL, NULL );
#endif  /* defined(PARSEC_PROF_TRACE) */

        return PARSEC_HOOK_RETURN_ASYNC;
    }
    gpu_task = progress_task;
    goto fetch_task_from_shared_queue;

 disable_gpu:
    /* Something wrong happened. Push all the pending tasks back on the
     * cores, and disable the gpu.
     */
    parsec_warning("Critical issue related to the GPU discovered. Giving up\n");
    return PARSEC_HOOK_RETURN_DISABLE;
}

#endif /* PARSEC_HAVE_CUDA */
