#include "timer.h"
#include <cstring>

namespace utils {

int parseLine( char* line )
{
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen( line );
    const char* p = line;
    while ( *p < '0' || *p > '9' )
        p++;
    line[i - 3] = '\0';
    i = atoi( p );
    return i;
}

int getValue(const int v) {
  // Note: this value is in KB!
  FILE *file = fopen("/proc/self/status", "r");
  int result = -1;
  char line[128];

  while (fgets(line, 128, file) != NULL) {
    switch (v) {
    case 0: {
      if (strncmp(line, "VmSize:", 7) == 0) {
        result = parseLine(line);
        break;
      }
      break;
    }
    case 1: {
      if (strncmp(line, "VmRSS:", 6) == 0) {
        result = parseLine(line);
        break;
      }
      break;
    }
    default:
      break;
    }
  }
  fclose(file);
  return result;
}
} // namespace utils