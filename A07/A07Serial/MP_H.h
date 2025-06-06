#ifndef MP_H
#define MP_H

#include <assert.h>
#include <stdlib.h>

#ifndef MP_ASSERT
    #define MP_ASSERT(x) assert(x)
#endif 

#ifndef MP_ALLOC
    #define MP_ALLOC(size) malloc(size)
#endif

#ifndef MP_FREE
    #define MP_FREE(ptr) free(ptr)
#endif

#endif // MP_H
