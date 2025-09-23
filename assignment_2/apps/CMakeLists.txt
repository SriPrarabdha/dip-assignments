add_executable(run_exp
    main.cpp
    exp1.cpp
    exp2.cpp
    exp3.cpp
)

target_include_directories(run_exp
    PRIVATE ${PROJECT_SOURCE_DIR}/apps/include
)

target_link_libraries(run_exp
    PRIVATE imageproc
)



# add_executable(run_exp main.cpp)
# target_link_libraries(run_exp PRIVATE imageproc)


