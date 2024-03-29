### Look for OMEGA
option(PARSEC_Q2J
  "Build the JDF generator" ON)
if(PARSEC_Q2J)
  find_package(Omega)
endif(PARSEC_Q2J)

if(OMEGA_FOUND)
  include_directories(${OMEGA_INCLUDE_DIRS})


  BISON_TARGET(parsec_q2j_yacc src/q2j.y ${CMAKE_CURRENT_BINARY_DIR}/q2j.y.c)
  FLEX_TARGET(parsec_q2j_flex src/q2j.l  ${CMAKE_CURRENT_BINARY_DIR}/q2j.l.c)
  ADD_FLEX_BISON_DEPENDENCY(parsec_q2j_flex parsec_q2j_yacc)

  # Bison and Flex are supposed to generate good code.
  # But they don't.
  # This approach is damageable, because we can't catch C errors in our .l or .y code
  # But if we don't do that, we'll keep having reports of compilation warnings forever.
  SET_SOURCE_FILES_PROPERTIES(${BISON_parsec_q2j_yacc_OUTPUTS} PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} -w")
  SET_SOURCE_FILES_PROPERTIES(${FLEX_parsec_q2j_flex_OUTPUTS} PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} -w")

  include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ptg-compiler)

  add_executable(q2j
      src/omega_interface.cpp
      src/jdfoutput.cpp
      src/jdfregister.cpp
      src/driver.c
      src/utility.c
      src/symtab.c
      ../ptg-compiler/jdf_unparse.c
      ${BISON_parsec_q2j_yacc_OUTPUTS}
      ${FLEX_parsec_q2j_flex_OUTPUTS}
  )
  set_target_properties(q2j PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(q2j m parsec-base ${OMEGA_LIBRARIES})

  install(TARGETS q2j RUNTIME DESTINATION bin)
endif(OMEGA_FOUND)
