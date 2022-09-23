//
// Created by mas on 9/23/22.
//

#ifndef BS_LOGGER_H
#define BS_LOGGER_H

enum LoggingLevel {
    BS_LOG_LEVEL_DEBUG,
    BS_LOG_LEVEL_INFO,
    BS_LOG_LEVEL_WARN,
    BS_LOG_LEVEL_ERROR,
    BS_LOG_LEVEL_FATAL,
};

__attribute__((unused)) void set_logging_level(enum LoggingLevel l);
__attribute__((unused)) void log_fatal(const char *msg);
__attribute__((unused)) void log_error(const char *msg);
__attribute__((unused)) void log_warn(const char *msg);
__attribute__((unused)) void log_info(const char *msg);
__attribute__((unused)) void log_debug(const char *msg);

#endif //BS_LOGGER_H

#ifdef BS_LOGGER_H_IMPL

#include <stdio.h>

#define RED   "\x1B[31m"
#define GREEN "\x1B[32m"
#define ORANGE "\x1B[33m"
#define BLUE "\x1B[34m"
#define BOLD "\x1B[1m"
#define RESET "\x1B[0m"

enum LoggingLevel level = BS_LOG_LEVEL_INFO;

__attribute__((unused)) void set_logging_level(enum LoggingLevel l) {
    level = l;
}


__attribute__((unused)) void log_debug(const char *msg)
{
    if (level <= BS_LOG_LEVEL_DEBUG)
    {
        printf("%s[DEBUG]%s: %s", BLUE, RESET, msg);
    }
}

__attribute__((unused)) void log_info(const char *msg)
{
    if (level <= BS_LOG_LEVEL_INFO)
    {
        printf("%s[INFO]%s: %s", GREEN, RESET, msg);
    }
}

__attribute__((unused)) void log_warn(const char *msg)
{
    if (level <= BS_LOG_LEVEL_WARN)
    {
        printf("%s[WARN]%s: %s", ORANGE, RESET, msg);
    }
}

__attribute__((unused)) void log_error(const char *msg)
{
    if (level <= BS_LOG_LEVEL_ERROR)
    {
        printf("%s[ERROR]%s: %s", RED, RESET, msg);
    }
}

__attribute__((unused)) void log_fatal(const char *msg)
{
    if (level <= BS_LOG_LEVEL_FATAL)
    {
        printf("%s%s[FATAL]%s: %s", RED, BOLD, RESET, msg);
    }
}

#endif //BS_LOGGER_H_IMPL
