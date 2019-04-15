include(PrecisionGenerator)

macro(testings_addexec OUTPUTLIST PRECISIONS ZSOURCES)
  include_directories(. ${COREBLAS_INCLUDE_DIRS})

  # Set flags for compilation
  if( MPI_C_FOUND )
    set(testings_addexec_CFLAGS  "${MPI_C_COMPILE_FLAGS} ${testings_addexec_CFLAGS}")
    set(testings_addexec_LDFLAGS "${MPI_C_LINK_FLAGS} ${testings_addexec_LDFLAGS}")
  endif( MPI_C_FOUND )
  set(testings_addexec_LIBS
      common dplasma
      ${EXTRA_LIBS})

  set(testings_addexec_GENFILES "")
  precisions_rules_py(testings_addexec_GENFILES
    "${ZSOURCES}"
    PRECISIONS "${PRECISIONS}")
  foreach(testings_addexec_GENFILE ${testings_addexec_GENFILES})
    string(REGEX REPLACE "\\.c" "" testings_addexec_EXEC ${testings_addexec_GENFILE})

    add_executable(${testings_addexec_EXEC} ${testings_addexec_GENFILE})
    add_dependencies(${testings_addexec_EXEC} dplasma dplasma_includes)
    if( PLASMA_F_COMPILE_SUCCESS )
      set_target_properties(${testings_addexec_EXEC} PROPERTIES
                              LINKER_LANGUAGE Fortran
                              COMPILE_FLAGS "${testings_addexec_CFLAGS}"
                              LINK_FLAGS "${testings_addexec_LDFLAGS} ${LOCAL_FORTRAN_LINK_FLAGS} ${COREBLAS_LDFLAGS}")
    else( PLASMA_F_COMPILE_SUCCESS )
      set_target_properties(${testings_addexec_EXEC} PROPERTIES
                              COMPILE_FLAGS "${testings_addexec_CFLAGS}"
                              LINK_FLAGS "${testings_addexec_LDFLAGS} ${COREBLAS_LDFLAGS}")
    endif( PLASMA_F_COMPILE_SUCCESS )
    target_link_libraries(${testings_addexec_EXEC} ${testings_addexec_LIBS} ${COREBLAS_LIBRARIES})
    #    install(TARGETS ${testings_addexec_EXEC} RUNTIME DESTINATION bin)
    list(APPEND ${OUTPUTLIST} ${testings_addexec_EXEC})
  endforeach()

endmacro(testings_addexec)

