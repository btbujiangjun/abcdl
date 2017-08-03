/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-08-02 11:05
* Last modified: 2017-08-02 11:05
* Filename: Logging.h
* Description: 
**********************************************/

#ifndef ABCDL_UTILS_LOGGING_H_
#define ABCDL_UTILS_LOGGING_H_

#include <sstream>

namespace abcdl{
namespace utils{
namespace logging{

//Log Levels
const int INFO = 0;
const int WARNING = 1;
const int ERROR = 2;
const int FATAL = 3;
const int NUM_SEVERITIES = 4;

#define _ABCDL_LOG_INFO \
    LogMessage(__FILE__, __LINE__, abcdl::utils::logging::INFO)

#define _ABCDL_LOG_WARNING \
    LogMessage(__FILE__, __LINE__, abcdl::utils::logging::WARNING)

#define _ABCDL_LOG_ERROR \
    LogMessage(__FILE__, __LINE__, abcdl::utils::logging::ERROR)

#define _ABCDL_LOG_FATAL \
    LogMessage(__FILE__, __LINE__, abcdl::utils::logging::FATAL)

#define ABCDL_LOG(severity) _ABCDL_LOG_##severity
#define LOG(severity) ABCDL_LOG(severity)

#define PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#define PREDICT_FALSE(x) (__builtin_expect(x, 0))

#define ABCDL_CHECK(condition) \
    if(PREDICT_FALSE(!(condition))) \
        ABCDL_LOG(FATAL) << "Check failed:" #condition " "

#define ABCDL_CHECK_EQ(v1, v2) ABCDL_CHECK((v1) == (v2))
#define ABCDL_CHECK_NE(v1, v2) ABCDL_CHECK((v1) != (v2))
#define ABCDL_CHECK_LE(v1, v2) ABCDL_CHECK((v1) <= (v2))
#define ABCDL_CHECK_LT(v1, v2) ABCDL_CHECK((v1) < (v2))
#define ABCDL_CHECK_GE(v1, v2) ABCDL_CHECK((v1) >= (v2))
#define ABCDL_CHECK_GT(v1, v2) ABCDL_CHECK((v1) > (v2))
#define ABCDL_CHECK_NOTNULL(v) ABCDL_CHECK((v) != NULL)

void initialize_logging(int argc, char** argv);

void set_min_log_level(int level);

void install_failure_function(void (*callback)());

void install_failure_writer(void(*callback)(const char*, int));

class LogMessage : public std::basic_ostringstream<char>{
public:
    LogMessage(const char* name,
               int line,
               int severity);

    ~LogMessage();

protected:
    void generate_log_message();

private:
    const char* _name;
    int _line;
    int _severity;
};//class LogMessage

class LogMessageFatal : public LogMessage{
public:
    LogMessageFatal(const char* file, int line) __attribute__((cold));

    ~LogMessageFatal() __attribute__((noreturn));
};//class LogMessageFatal


}//end namespace logging
}//end namespace utils
}//end namespace abcdl
#endif //ABCDL_UTILS_LOGGING_H_
