//
// Created by wyz on 2021/6/10.
//
#include"IO/reader_impl.hpp"

VS_START


std::unique_ptr<Reader> Reader::CreateReader() {
    return std::unique_ptr<ReaderImpl>();
}





void ReaderImpl::GetPacket(const std::array<uint32_t, 4> &idx, std::vector<std::vector<uint8_t>> &packet) {
    readers[idx[3]]->read_packet({idx[0],idx[1],idx[2]},packet);
}

void ReaderImpl::AddLodData(int lod, const char *path) {
    readers[lod]=std::make_unique<sv::Reader>(path);
}

VS_END