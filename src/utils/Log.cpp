/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-08-02 11:25
* Last modified: 2017-08-03 15:06
* Filename: Logging.cpp
* Description: 
**********************************************/

#include "utils/Log.h"
#include "stdlib.h"
#include "stdio.h"
#include "unistd.h"
#include "fcntl.h"
#include "string.h"
#include <time.h>
#include <vector>

namespace abcdl{
namespace utils{
namespace log{

static std::string join(const std::string& part1, const std::string& part2){
    const char sep = '/';
    if(!part2.empty() && part2.front() == sep){
        return part2;
    }

    std::string return_value;
    return_value.reserve(part1.size() + part2.size() + 1);
    return_value = part1;
    if(!return_value.empty() && return_value.back() != sep){
        return_value += sep;
    }
    return_value += part2;

    return return_value;
}

static inline bool env2bool(const char* env_name, bool default_value = false){
    char* env_value = getenv(env_name);
    if(env_value == nullptr){
        return default_value;
    }else{
        return memchr("tTyY1\0", env_value[0], 6) != nullptr;
    }
}
static inline int env2int(const char* env_name, int default_value = 0){
    char* env_value = getenv(env_name);
    if(env_value == nullptr){
        return default_value;
    }
        
    int return_value = default_value;
    try{
        return_value = std::stoi(env_value);
    }catch(...){
        //pass
    }
    return return_value;
}

static inline int env2index(const char* env_name,
                            const std::vector<std::string>& options,
                            int default_value){
    char* env_value = getenv(env_name);
    if(env_value != nullptr){
        for(size_t i = 0; i < options.size(); i++){
            if(options[i] == env_value){
                return static_cast<int>(i);
            }
        }
    }
    return default_value;
}


static bool g_log_inited = false;
static bool g_log_to_stderr = env2bool("ABCDL_LOG_LOGTOSTDERR", true);
static const std::vector<std::string> g_log_level_name = {"INFO", "WARNING", "ERROR", "FATAL"};
static int g_log_min_level = env2int("ABCDL_LOG_MIN_LEVEL", env2index("ABCDL_LOG_MIN_LEVEL", g_log_level_name, 0));

static std::vector<std::vector<int>> g_log_fds;
static std::vector<int> g_log_file_fds;

static void (*g_failure_function_ptr)() __attribute__((noreturn)) = abort;

static void free_log_file_fds(){
    for(auto fd : g_log_file_fds){
        close(fd);
    }
}
static void initialize_log_fds(char* argc){
    g_log_fds.resize(NUM_SEVERITIES);
    for(int i = g_log_min_level; i < NUM_SEVERITIES && g_log_to_stderr; i++){
        std::vector<int>& fds = g_log_fds[i];
        fds.push_back(STDERR_FILENO);
    }
    char* log_dir = getenv("ABCDL_LOG_LOGDIR");
    if(log_dir == nullptr){
        char path = '.';
        log_dir = &path;
    }

    for(int i = g_log_min_level; i != NUM_SEVERITIES && log_dir != nullptr; i++){
        std::string file_name = join(log_dir, std::string(argc) + "." + g_log_level_name[i]);
        int fd = open(file_name.c_str(), O_CREAT | O_WRONLY, 0644);
        if( fd == -1){
            fprintf(stderr, "Open log file error!");
            exit(-1);
        }
        g_log_file_fds.push_back(fd);
        std::vector<int>& current_fds = g_log_fds[i];
        current_fds.insert(current_fds.end(), g_log_file_fds.begin(), g_log_file_fds.end());
    }
    atexit(free_log_file_fds);
    g_log_inited = true;
}


void initialize_log(int argc, char** argv){
    initialize_log_fds(argv[0]);
}

void set_min_log_level(int level){
    g_log_min_level = level;
}

void install_failure_function(void (*callback)()){
    g_failure_function_ptr = callback;
}

void install_failure_writer(void(*callback)(const char*, int)){
    (void)(callback);
}

LogMessage::LogMessage(const char* name,
                       int line,
                       int severity) : 
    _name(name), _line(line), _severity(severity){}

LogMessage::~LogMessage(){
    this->generate_log_message();
}

void LogMessage::generate_log_message(){
    time_t timep;
    time(&timep);
    char time_data[64];
    strftime(time_data, sizeof(time_data), "%Y-%m-%d %H:%M:%S", localtime(&timep));
    if(!g_log_inited){
        fprintf(stderr, "[%s] [%s] [%s:%d] %s\n", g_log_level_name[_severity].c_str(), time_data,  _name, _line, str().c_str());
    }else{
        for(auto& fd : g_log_fds[this->_severity]){
            dprintf(fd, "[%s] [%s] [%s:%d] %s\n", g_log_level_name[_severity].c_str(), time_data, _name, _line, str().c_str());
        }
    }
}


LogMessageFatal::LogMessageFatal(const char* file, int line) : 
    LogMessage(file, line, FATAL){}

LogMessageFatal::~LogMessageFatal(){
    generate_log_message();
    g_failure_function_ptr();
}

}//namespace log
}//namespace utils
}//namespace abcdl
