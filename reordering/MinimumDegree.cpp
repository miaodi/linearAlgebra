#include "MinimumDegree.hpp"
#include <algorithm>
#include <assert.h>
namespace reordering {

template <typename COLTYPE>
template <typename ROWTYPE>
void QuotientGraph<COLTYPE>::initialize(const COLTYPE nnodes, ROWTYPE const *ai,
                                        COLTYPE const *aj) {
  _nodes.resize(nnodes);
  _union_find.reset(nnodes); // reset union-find structure

  _degree_to_principle.clear();

  __temp1.reserve(nnodes);
  __temp2.reserve(nnodes); // temporary vectors for sorting

  const ROWTYPE base = ai[0];
  for (COLTYPE i = 0; i < nnodes; i++) {
    _nodes[i].adjacent_variables.reserve(ai[i + 1] - ai[i]);
    for (ROWTYPE j_idx = ai[i] - base; j_idx < ai[i + 1] - base; j_idx++) {
      COLTYPE j = aj[j_idx] - base;
      if (j == i)
        continue;
      _nodes[i].adjacent_variables.push_back(j);
    }
    _nodes[i].degree = _nodes[i].adjacent_variables.size();
    _degree_to_principle[_nodes[i].degree].insert(i);
    _nodes[i].simple_variables.push_back(i);
  }
}

template <typename COLTYPE>
template <typename ROWTYPE>
void QuotientGraph<COLTYPE>::operator()(const COLTYPE nnodes, ROWTYPE const *ai,
                                        COLTYPE const *aj, COLTYPE *iperm,
                                        COLTYPE *perm) {
  initialize(nnodes, ai, aj);
}

template <typename COLTYPE>
bool QuotientGraph<COLTYPE>::isDistinguishable(const COLTYPE i,
                                               const COLTYPE j) const {
  // diff in quotient graph
  if (_nodes[i].adjacent_variables.size() !=
      _nodes[j].adjacent_variables.size())
    return true;
  if (_nodes[i].adjacent_elements.size() != _nodes[j].adjacent_elements.size())
    return true;

  // check if they have the same adjacent variables
  size_t k = 0, l = 0;
  while (k < _nodes[i].adjacent_variables.size() &&
         l < _nodes[j].adjacent_variables.size()) {
    if (_nodes[i].adjacent_variables[k] != _nodes[j].adjacent_variables[l]) {
      if (_nodes[i].adjacent_variables[k] == j)
        k++;
      else if (_nodes[j].adjacent_variables[l] == i)
        l++;
      else
        return true; // different neighbours
    } else {
      k++;
      l++;
    }
  }
  if (k < _nodes[i].adjacent_variables.size()) {
    if (_nodes[i].adjacent_variables[k] == j)
      k++;
    else
      return true; // different neighbours
  }
  if (l < _nodes[j].adjacent_variables.size()) {
    if (_nodes[j].adjacent_variables[l] == i)
      l++;
    else
      return true; // different neighbours
  }
  if (k != _nodes[i].adjacent_variables.size() ||
      l != _nodes[j].adjacent_variables.size())
    return true; // different neighbours

  // check if they have the same adjacent elements
  k = 0;
  l = 0;
  while (k < _nodes[i].adjacent_elements.size() &&
         l < _nodes[j].adjacent_elements.size()) {
    if (_nodes[i].adjacent_elements[k] != _nodes[j].adjacent_elements[l]) {
      return true;
    } else {
      k++;
      l++;
    }
  }
  if (k < _nodes[i].adjacent_elements.size() ||
      l < _nodes[j].adjacent_elements.size())
    return true; // different neighbours
  return false;  // distinguishable
}

template <typename COLTYPE>
void QuotientGraph<COLTYPE>::updateNode(const COLTYPE i,
                                        std::vector<COLTYPE> &temp) {
  assert(i == _union_find.Find(i));
  principleVector(_nodes[i].simple_variables, temp);
  principleVector(_nodes[i].adjacent_variables, temp);
  principleVector(_nodes[i].adjacent_elements, temp);
}

template <typename COLTYPE>
void QuotientGraph<COLTYPE>::getFillins(const COLTYPE i,
                                        std::vector<COLTYPE> &temp1,
                                        std::vector<COLTYPE> &temp2) {
  // TODO: check if really needed
  updateNode(i, temp2);

  temp1.clear();
  __vectors.clear();
  __vectors.push_back(&_nodes[i].adjacent_variables);

  for(auto j: _nodes[i].adjacent_elements) {
    updateNode(j, temp2);
    __vectors.push_back(&_nodes[j].adjacent_variables);
  }

  mergeKVectors(__vectors, temp1);
  vectorSubtract(temp1, _nodes[i].simple_variables);
}

template <typename COLTYPE>
void QuotientGraph<COLTYPE>::merge(const COLTYPE i, const COLTYPE j) {
  assert(_union_find.Find(i) !=
         _union_find.Find(j)); // cannot merge the same element

  _union_find.Unite(i, j);
  std::vector<COLTYPE> temp(_nodes[i].simple_variables.size() +
                            _nodes[j].simple_variables.size());
  auto it = std::set_union(_nodes[i].simple_variables.begin(),
                           _nodes[i].simple_variables.end(),
                           _nodes[j].simple_variables.begin(),
                           _nodes[j].simple_variables.end(), temp.begin());
  std::swap(_nodes[i].simple_variables, temp);
  _nodes[i].degree -= static_cast<COLTYPE>(_nodes[j].simple_variables.size());
  _nodes[j].simple_variables.clear();
  _nodes[j].adjacent_elements.clear();
  _nodes[j].adjacent_variables.clear();
}

template <typename COLTYPE>
void QuotientGraph<COLTYPE>::massElimination(const COLTYPE p) {
  getFillins(p, __temp1, __temp2);
  auto &fill_ins = __temp1;
  for (auto i : fill_ins) {
    // remove redundant variables
    vectorSubtract(_nodes[i].adjacent_variables, fill_ins);
    vectorSubtract(_nodes[i].adjacent_variables, _nodes[p].simple_variables);

    // element absorption
    vectorSubtract(_nodes[i].adjacent_elements, _nodes[p].adjacent_elements);
    _nodes[i].adjacent_elements.insert(
        std::upper_bound(_nodes[i].adjacent_elements.begin(),
                         _nodes[i].adjacent_elements.end(), p),
        p);
    _nodes[i].degree = getExternalDegree(i);
  }
}

template <typename COLTYPE>
void QuotientGraph<COLTYPE>::supervariableMerge(
    const std::vector<COLTYPE> &fillins) {
  std::map<COLTYPE, std::vector<COLTYPE>> supervariables;
  for (auto i : fillins) {
    auto [it, inserted] = supervariables.emplace(i, std::vector<COLTYPE>());
    it->second.push_back(i);
  }
  for (const auto &pair : supervariables) {
    const auto &i = pair.first;
    const auto &fillin = pair.second;
    for (size_t j = 0; j < fillin.size(); ++j) {
      for (size_t k = j + 1; k < fillin.size(); ++k) {
        auto j_elem = fillin[j];
        auto k_elem = fillin[k];
        if (isDistinguishable(j_elem, k_elem)) {
          continue; // cannot merge
        }
        merge(j_elem, k_elem);

      }
    }
  }
}

template <typename COLTYPE>
COLTYPE QuotientGraph<COLTYPE>::getExternalDegree(const COLTYPE i) {
  getFillins(i, __temp2, __temp3);
  return static_cast<COLTYPE>(__temp2.size());
}

template <typename COLTYPE>
void QuotientGraph<COLTYPE>::principleVector(std::vector<COLTYPE> &vec,
                                             std::vector<COLTYPE> &temp) {
  temp.clear();
  for (auto i : vec) {
    temp.push_back(_union_find.Find(i));
  }
  std::sort(temp.begin(), temp.end());
  vec.clear();
  for (auto i : temp) {
    if (vec.empty() || vec.back() != i) {
      vec.push_back(i);
    }
  }
}

template class QuotientGraph<int>;
template void QuotientGraph<int>::operator()(const int nnodes, int const *ai,
                                             int const *aj, int *iperm,
                                             int *perm);

template void QuotientGraph<int>::initialize(const int nnodes, int const *ai,
                                             int const *aj);

} // namespace reordering