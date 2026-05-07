include_guard(GLOBAL)

function(linear_algebra_set_default_cache_dir cache_variable cache_leaf help_text)
    if(DEFINED ${cache_variable} AND NOT "${${cache_variable}}" STREQUAL "")
        return()
    endif()

    if(DEFINED ENV{XDG_CACHE_HOME} AND NOT "$ENV{XDG_CACHE_HOME}" STREQUAL "")
        set(_cache_root "$ENV{XDG_CACHE_HOME}")
    elseif(DEFINED ENV{HOME} AND NOT "$ENV{HOME}" STREQUAL "")
        set(_cache_root "$ENV{HOME}/.cache")
    else()
        set(_cache_root "${CMAKE_BINARY_DIR}/.cache")
    endif()

    set(${cache_variable}
        "${_cache_root}/linearAlgebra/${cache_leaf}"
        CACHE PATH "${help_text}" FORCE)
endfunction()

function(linear_algebra_prefer_user_python)
    if(DEFINED Python3_EXECUTABLE AND NOT "${Python3_EXECUTABLE}" STREQUAL "")
        return()
    endif()

    if(DEFINED ENV{PYTHONWITHSSGETPY} AND NOT "$ENV{PYTHONWITHSSGETPY}" STREQUAL "")
        set(_python_with_ssgetpy "$ENV{PYTHONWITHSSGETPY}")
        if(NOT EXISTS "${_python_with_ssgetpy}")
            message(FATAL_ERROR
                "PYTHONWITHSSGETPY is set to '${_python_with_ssgetpy}', but that file does not exist")
        endif()

        set(Python3_EXECUTABLE "${_python_with_ssgetpy}"
            CACHE FILEPATH "Python interpreter used for SuiteSparse matrix downloads" FORCE)
        return()
    endif()

    if(DEFINED ENV{HOME} AND EXISTS "$ENV{HOME}/.venv/bin/python")
        set(Python3_EXECUTABLE "$ENV{HOME}/.venv/bin/python"
            CACHE FILEPATH "Python interpreter used for SuiteSparse matrix downloads" FORCE)
    endif()
endfunction()
