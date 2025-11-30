/*
 * BSD 2-Clause License
 * 
 * Copyright (c) 2020, Alessandro Capotondi
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * 
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * @file utils.c
 * @author Alessandro Capotondi
 * @date 27 Mar 2020
 * @brief File containing utilities functions for HPC Unimore Class
 *
 * Utilities for OpenMP lab.
 * 
 * @see http://algo.ing.unimo.it/people/andrea/Didattica/HPC/index.html
 */

/*  only timing functions and a print */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif
#include <time.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

static struct timespec utils_ts_start;
static struct timespec utils_ts_end;

static unsigned long long diff_ns(const struct timespec *start, const struct timespec *end)
{
    long sec = end->tv_sec - start->tv_sec;
    long nsec = end->tv_nsec - start->tv_nsec;
    if (nsec < 0)
    {
        sec -= 1;
        nsec += 1000000000L;
    }
    return (unsigned long long)sec * 1000000000ULL + (unsigned long long)nsec;
}

void start_timer(void)
{
    clock_gettime(CLOCK_MONOTONIC, &utils_ts_start);
}

void stop_timer(void)
{
    clock_gettime(CLOCK_MONOTONIC, &utils_ts_end);
}

unsigned long long elapsed_ns(void)
{
    return diff_ns(&utils_ts_start, &utils_ts_end);
}

void print_elapsed_ms(const char *label)
{
    unsigned long long ns = elapsed_ns();
    double ms = (double)ns / 1.0e6;
    if (label)
        printf("%s: %.3f ms\n", label, ms);
    else
        printf("Elapsed: %.3f ms\n", ms);
}

#ifdef __cplusplus
}
#endif
