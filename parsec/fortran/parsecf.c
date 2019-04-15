/*
 * Copyright (c) 2013-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec/parsec_config.h"
#include "parsec/runtime.h"
#include <string.h>
#include <stdlib.h>

void parsec_init_f08(int nbcores, parsec_context_t** context, int* ierr)
{
    char *args, *token, **argv = NULL;
    int argc = 0;

    if( NULL != (args = getenv("PARSEC_ARGS"))) {
        args = token = strdup(args);
        while(NULL != strsep(&args, ";=")) argc++;
        argv = (char**)malloc((2+argc) * sizeof(char*));
        free(token);
        args = strdup(getenv("PARSEC_ARGS"));
        argc = 1;
        argv[0] = "myapp";  /* No idea how to extract the real application name from Fortran */
        while( NULL != (token = strsep(&args, ";=")) ) {
            argv[argc] = token;
            argc++;
        }
        argv[argc] = NULL;
    }
    *context = parsec_init(nbcores, &argc, &argv);
    free(argv);
    *ierr = (NULL == *context) ? 0 : -1;
}

void parsec_fini_f08(parsec_context_t** context, int* ierr)
{
    *ierr = parsec_fini(context);
}

void
parsec_taskpool_set_complete_callback_f08(parsec_taskpool_t** tp,
                                          parsec_event_cb_t complete_cb,
                                          void* cb_data, int* ierr)
{
    *ierr = parsec_taskpool_set_complete_callback(*tp, complete_cb, cb_data);
}

void
parsec_taskpool_get_complete_callback_f08(parsec_taskpool_t** tp,
                                          parsec_event_cb_t* complete_cb,
                                          void** cb_data, int* ierr)
{
    *ierr = parsec_taskpool_get_complete_callback(*tp, complete_cb, cb_data);
}

void
parsec_taskpool_set_enqueue_callback_f08(parsec_taskpool_t** tp,
                                         parsec_event_cb_t enqueue_cb,
                                         void* cb_data, int* ierr)
{
    *ierr = parsec_taskpool_set_enqueue_callback(*tp, enqueue_cb, cb_data);
}

void
parsec_taskpool_get_enqueue_callback_f08(parsec_taskpool_t** tp,
                                         parsec_event_cb_t* enqueue_cb,
                                         void** cb_data, int* ierr)
{
    *ierr = parsec_taskpool_get_enqueue_callback(*tp, enqueue_cb, cb_data);
}

void
parsec_taskpool_set_priority_f08(parsec_taskpool_t** tp, int priority, int* ierr)
{
    *ierr = parsec_taskpool_set_priority(*tp, priority);
}

