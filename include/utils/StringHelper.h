/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-06-26 11:59
 * Last modified : 2017-06-26 11:59
 * Filename      : StringHelper.h
 * Description   : 
 *         split,       // split std::string by delim(such as " .,/")
 *         int2str,     // convert int to std::string
 *         real2str,   // convert real to std::string
 *         str2int,     // convert std::string to int
 *         str2real,   // convert std::string to real
 *         strtoupper,  // all to upper case
 *         strtolower,  // all to lower case
 **********************************************/
#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <regex>
#include "utils/TypeDef.h"

namespace abcdl{
namespace utils{
class StringHelper{
public:
    std::vector<std::string> split(const std::string& str, const std::string& split);

    std::string int2str(int n);
    std::string real2str(real f);

    int str2int(std::string& s);
    real str2real(std::string& s);

    template<class Type>
    std::string tostring(Type a);

    template<class ToType,class FromType>
    ToType strconvert(FromType t);

    std::string& strtolower(std::string& s);
    std::string strtolower(std::string s);

    std::string& strtoupper(std::string& s);
    std::string strtoupper(std::string s);

};//class StringHelper

/**
 * @brief split a std::string by delim
 *
 * @param str std::string to be splited
 * @param c delimiter, const char*, just like " .,/", white space, dot, comma, splash
 *
 * @return a std::string vector saved all the splited world
 */
std::vector<std::string> StringHelper::split(const std::string& str, const std::string& split){
	std::regex re(split);
	std::sregex_token_iterator first{str.begin(), str.end(), re, -1}, last;
	return {first, last};
}

/**
 * @brief convert a integer into std::string through std::stringstream
 *
 * @param n a integer
 *
 * @return the std::string form of n
 */
std::string StringHelper::int2str(int n){
    std::stringstream ss;
    std::string s;
    ss << n;
    ss >> s;
    return s;
}

std::string StringHelper::real2str(real f){
    std::stringstream ss;
    std::string s;
    ss << f;
    ss >> s;
    return s;
}

/**
 * @brief convert something to std::string form through std::stringstream
 *
 * @tparam Type Type can be int,real
 * @param a 
 *
 * @return the std::string form of param a  
 */
template<class Type>
std::string StringHelper::tostring(Type a){
    std::stringstream ss;
    std::string s;
    ss << a;
    ss >> s;
    return s;
}

/**
 * @brief convert std::string to int by atoi
 *
 * @param s std::string
 *
 * @return the integer result
 */
int StringHelper::str2int(std::string& s){
    return atoi(s.c_str());
}

real StringHelper::str2real(std::string& s){
    return atof(s.c_str());
}

/**
 * @brief do std::string convert through std::stringstream from FromType to ToType
 *
 * @tparam ToType target type
 * @tparam FromType source type
 * @param t to be converted param
 *
 * @return the target form of param t
 */
template<class ToType,class FromType>
ToType StringHelper::strconvert(FromType t){
    std::stringstream ss;
    ToType a;
    ss << t;
    ss >> a;
    return a;
}

/**
 * @brief convert std::string to upper case throught transform method, also can use transform method directly
 *
 * @param s
 *
 * @return the upper case result saved still in s
 */
std::string& StringHelper::strtoupper(std::string& s){
    transform(s.begin(), s.end(), s.begin(), ::toupper);
    return s;
}

/**
 * @brief convert std::string to upper case through toupper, which transform a char into upper case 
 *
 * @param s
 *
 * @return the upper case result std::string
 */
std::string StringHelper::strtoupper(std::string s){
    std::string t = s;
    int i = -1;
    while(t[i++]){
        t[i] = toupper(t[i]);
    }
    return t;
}

/**
 * @brief convert std::string to lower case throught transform method, also can use transform method directly
 *
 * @param s
 *
 * @return the lower case result saved still in s
 */
std::string& StringHelper::strtolower(std::string& s){
    transform(s.begin(),s.end(),s.begin(),::tolower);
    return s;
}

/**
 * @brief convert std::string to lower case through tolower, which transform a char into lower case 
 *
 * @param s
 *
 * @return the lower case result std::string
 */
std::string StringHelper::strtolower(std::string s){
    std::string t = s;
    int i = -1;
    while(t[i++]){
        t[i] = tolower(t[i]);
    }
    return t;
}

}//namespace utils
}//namespace abcdl
