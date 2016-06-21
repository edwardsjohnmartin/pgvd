if (OPENCL_FOUND)
  add_subdirectory(src)
  add_subdirectory(include)
  set ( CLFW_INCLUDE_DIR
    ${CLFW_INCLUDE_DIR}
    PARENT_SCOPE
  )
else (OPENCL_FOUND)
  message(FATAL_ERROR "CLFW: OpenCL package is required")
endif (OPENCL_FOUND)