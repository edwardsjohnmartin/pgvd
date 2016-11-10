/***************************************************
** Min Surface Project                            **
** Copyright (c) 2014 University of Utah          **
** Scientific Computing and Imaging Institute     **
** 72 S Central Campus Drive, Room 3750           **
** Salt Lake City, UT 84112                       **
**                                                **
** For information about this project contact     **
** Valerio Pascucci at pascucci@sci.utah.edu      **
**                                                **
****************************************************/

#ifndef __TIMER_H__
#define __TIMER_H__

#include <time.h>
#include <sstream>
#include <string>
#include <iostream>

#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

class Timer {
 public:
  Timer(log4cplus::Logger& logger,  // NOLINT(runtime/references)
        const std::string& msg = "", const std::string& id = "",
        bool initial_msg = false,
        // const log4cplus::LogLevel log_level = log4cplus::TRACE_LOG_LEVEL)
        const log4cplus::LogLevel log_level = log4cplus::INFO_LOG_LEVEL)
      : _logger(&logger), _msg(msg), _id(id), _t(clock()), _active(true),
        _initial_msg(initial_msg), _alive(true),
        // _log_level(log4cplus::DEBUG_LOG_LEVEL) {
        // _log_level(log4cplus::TRACE_LOG_LEVEL) {
        _log_level(log_level) {
    initial();
  }

  ~Timer() {
    stop();
  }

  void TraceLevel() {
    _log_level = log4cplus::TRACE_LOG_LEVEL;
  }

  void DebugLevel() {
    _log_level = log4cplus::DEBUG_LOG_LEVEL;
  }

  void InfoLevel() {
    _log_level = log4cplus::INFO_LOG_LEVEL;
  }

  void kill() {
    _alive = false;
  }

  void suspend() {
    if (!_alive) return;
    _active = false;
    _t = clock()-_t;
  }

  void resume() {
    if (!_alive) return;
    _active = true;
    _t = clock()-_t;
  }

  void restart(const std::string& msg) {
    if (!_alive) return;
    stop();
    _msg = msg;
    _t = clock();
    _active = true;
    initial();
  }

  void reset(const std::string& msg) {
    if (!_alive) return;
    restart(msg);
  }

  void stop() {
    if (!_alive) return;
    if (_active) {
      const double t = (clock()-_t)/static_cast<double>(CLOCKS_PER_SEC);
      std::stringstream ss;
      if (!_id.empty())
        ss << _id << " " << _msg << " "
           << t << " sec";
      else
        ss << _msg << " " << t << " sec";
      log(ss.str());
      _active = false;
    }
  }

 private:
  void initial() const {
    if (!_alive) return;
    if (_initial_msg) {
      std::stringstream ss;
      if (!_id.empty())
        ss << _id << " " << _msg << "...";
      else
        ss << _msg << "...";
      log(ss.str());
    }
  }

  void log(const std::string& msg) const {
    if (_log_level == log4cplus::TRACE_LOG_LEVEL) {
      LOG4CPLUS_TRACE(*_logger, msg);
    } else if (_log_level == log4cplus::DEBUG_LOG_LEVEL) {
      LOG4CPLUS_DEBUG(*_logger, msg);
    } else if (_log_level == log4cplus::INFO_LOG_LEVEL) {
      LOG4CPLUS_INFO(*_logger, msg);
    }
  }

 private:
  log4cplus::Logger* _logger;
  std::string _msg;
  std::string _id;
  time_t _t;
  bool _active;
  bool _initial_msg;
  bool _alive;
  log4cplus::LogLevel _log_level;
};

#endif
