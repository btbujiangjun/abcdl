/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-08-02 11:05
* Last modified: 2017-09-01 10:50
* Filename: Log.h
* Description: 
**********************************************/
#pragma once

#include <sstream>

namespace abcdl{
namespace utils{
namespace log{

const int INFO = 0;
const int WARNING = 1;
const int ERROR = 2;
const int FATAL = 3;
const int NUM_SEVERITIES = 4;

#define _ABCDL_LOG_INFO \
    abcdl::utils::log::LogMessage(__FILE__, __LINE__, abcdl::utils::log::INFO)

#define _ABCDL_LOG_WARNING \
    abcdl::utils::log::LogMessage(__FILE__, __LINE__, abcdl::utils::log::WARNING)

#define _ABCDL_LOG_ERROR \
    abcdl::utils::log::LogMessage(__FILE__, __LINE__, abcdl::utils::log::ERROR)

#define _ABCDL_LOG_FATAL \
    abcdl::utils::log::LogMessage(__FILE__, __LINE__, abcdl::utils::log::FATAL)

#define ABCDL_LOG(severity) _ABCDL_LOG_##severity
#define LOG(severity) ABCDL_LOG(severity)

#define PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#define PREDICT_FALSE(x) (__builtin_expect(x, 0))

#define CHECK(condition) \
    if(PREDICT_FALSE(!(condition))) \
        LOG(FATAL) << "Check failed:" #condition " "

#define CHECK_EQ(v1, v2) CHECK((v1) == (v2))
#define CHECK_NE(v1, v2) CHECK((v1) != (v2))
#define CHECK_LE(v1, v2) CHECK((v1) <= (v2))
#define CHECK_LT(v1, v2) CHECK((v1) < (v2))
#define CHECK_GE(v1, v2) CHECK((v1) >= (v2))
#define CHECK_GT(v1, v2) CHECK((v1) > (v2))
#define CHECK_NOTNULL(v) CHECK((v) != NULL)

void initialize_log(int argc, char** argv);

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

}//end namespace log
}//end namespace utils
}//end namespace abcdl

using namespace abcdl::utils::log;
