#ifndef __TIME_UTILS_H__
#define __TIME_UTILS_H__

#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#define START_TIME(x)			gettimeofday(&x->stime, NULL);
#define END_TIME(x)				do {                                                \
                                    gettimeofday(&x->etime, NULL);                  \
								    timersub(&x->stime, &x->etime, &x->res);        \
                                    x->total_sec += x->res.tv_sec;                  \
								    x->total_usec += x->res.tv_usec;                \
                                } while(0)

#define TOTAL_SEC_TIME(x)		(x->total_sec)
#define TOTAL_SEC_UTIME(x)		(x->total_usec)

typedef struct timeutils {
	struct timeval stime;
	struct timeval etime;
    struct timeval res;

	long total_sec;
	long total_usec;
} timeutils;

#endif /* __TIME_UTILS_H__ */
