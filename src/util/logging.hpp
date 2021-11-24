#ifndef LOGGING_HPP
#define LOGGING_HPP


#ifndef LOGGING_ENABLED

    #define log_debug(fmt, ...)
    #define log_info(fmt, ...)
    #define log_warn(fmt, ...)
    #define log_error(fmt, ...)

#else

    #ifdef USE_LOGURU

    #include <loguru.hpp>

    #define log_debug(fmt, ...) (LOG_F(1, fmt, ## __VA_ARGS__))
    #define log_info(fmt, ...) (LOG_F(INFO, fmt, ## __VA_ARGS__))
    #define log_warn(fmt, ...) (LOG_F(WARNING, fmt, ## __VA_ARGS__))
    #define log_error(fmt, ...) (LOG_F(ERROR, fmt, ## __VA_ARGS__))

    #else

    // ## is a "gcc" hack for allowing empty __VA_ARGS__
    // https://stackoverflow.com/questions/5891221/variadic-macros-with-zero-arguments
    #define log_debug(fmt, ...) ((void)printf(fmt"\n", ## __VA_ARGS__))
    #define log_info(fmt, ...) ((void)printf(fmt"\n", ## __VA_ARGS__))
    #define log_warn(fmt, ...) ((void)printf(fmt"\n", ## __VA_ARGS__))
    #define log_error(fmt, ...) ((void)printf(fmt"\n", ## __VA_ARGS__))

    #endif

#endif // NO_LOGGING

#endif // define LOGGING_HPP
