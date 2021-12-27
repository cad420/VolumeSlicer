
set(MPI_CXX_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/third_party/MPI/Include)

set(MPI_CXX_LIBRARIES
        ${PROJECT_SOURCE_DIR}/third_party/MPI/Lib/x64/msmpi.lib;
        ${PROJECT_SOURCE_DIR}/third_party/MPI/Lib/x64/msmpifec.lib;
        ${PROJECT_SOURCE_DIR}/third_party/MPI/Lib/x64/msmpifmc.lib)