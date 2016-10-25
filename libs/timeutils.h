#ifndef __TIME_UTILS_H__
#define __TIME_UTILS_H__

#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#define START_TIME(x)			gettimeofday(&x->stime, NULL);
#define END_TIME(x)				do {                                                \
                                    gettimeofday(&x->etime, NULL);                  \
								    timersub(&x->etime, &x->stime, &x->diff);       \
                                    timeradd(&x->diff, &x->total, &x->total);       \
                                } while(0)

#define TOTAL_SEC_TIME(x)		(x->total.tv_sec)
#define TOTAL_SEC_UTIME(x)		(x->total.tv_usec)

typedef struct timeutils {
	struct timeval stime;
	struct timeval etime;
    struct timeval diff;

    struct timeval total;

} timeutils;

#endif /* __TIME_UTILS_H__ */
