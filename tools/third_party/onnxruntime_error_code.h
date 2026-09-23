#ifndef ORT_ERROR_CODE_SHIM_H
#define ORT_ERROR_CODE_SHIM_H
#include <stdint.h>
typedef int32_t OrtErrorCode;
enum {
  ORT_OK = 0,
  ORT_FAIL = 1,
  ORT_INVALID_ARGUMENT = 2,
  ORT_NO_SUCHFILE = 3
};
#endif
