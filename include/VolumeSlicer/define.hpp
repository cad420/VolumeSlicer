//
// Created by wyz on 2021/6/9.
//

#pragma once

#define MAX_SLICE_W 10000
#define MAX_SLICE_H 10000

#define FLOAT_ZERO 0.001f

#define INVALID 0xffffffff

#define FORWARD_IMPL_DECLARATION(cls) \
class  cls##Impl

#define EXPLICT_INSTANCE_TEMPLATE_CLASS(CLS, T) template class CLS<T>;

#define EXPLICT_INSTANCE_TEMPLATE_TEMPLATE_CLASS(CLS, T, ...) template class CLS<T<__VA_ARGS__>>;

#define EXPLICT_INSTANCE_TEMPLATE_FUNCTION(T,RET,FUN,...) template RET FUN<T>(__VA_ARGS__);


