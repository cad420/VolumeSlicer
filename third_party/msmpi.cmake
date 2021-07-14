
set(MPI_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/third_party/MPI/Include)

set(MPI_LIBRARIES
        ${PROJECT_SOURCE_DIR}/third_party/MPI/Lib/x64/msmpi.lib;
        ${PROJECT_SOURCE_DIR}/third_party/MPI/Lib/x64/msmpifec.lib;
        ${PROJECT_SOURCE_DIR}/third_party/MPI/Lib/x64/msmpifmc.lib)