#include "matrix_utils.hpp"
#include <limits>

namespace matrix_utils {
template <typename ROWTYPE, typename COLTYPE>
void ElimTree(const COLTYPE rows, const int base, ROWTYPE const *ai,
              COLTYPE const *aj, COLTYPE *parent) {
  const COLTYPE empty_tag = std::numeric_limits<COLTYPE>::max();
  // initialize parent
  std::fill_n(parent, rows, empty_tag);

  COLTYPE jroot = empty_tag;
  for (COLTYPE i = 0; i < rows; i++) {
    for (ROWTYPE j = ai[i] - base, jroot = aj[j] - base;
         j < ai[i + 1] - base && jroot < i; j++) {
      while (parent[jroot] != empty_tag && parent[jroot] != i + base) {
        jroot = parent[jroot] - base;
      }
      if (parent[jroot] == empty_tag)
        parent[jroot] = i + base;
    }
  }
}
} // namespace matrix_utils