#include "MinimumDegree.hpp"
#include <algorithm>
#include <assert.h>
#include <limits> // Required for std::numeric_limits
namespace reordering
{

  template <typename COLTYPE>
  template <typename ROWTYPE>
  void QuotientGraph<COLTYPE>::initialize(const COLTYPE nnodes, ROWTYPE const *ai,
                                          COLTYPE const *aj)
  {
    using ObjectType = typename decltype(_pool)::value_type;
    _pool.setObjectPrep([nnodes](ObjectType *obj)
                        {obj->reserve(nnodes/10.);obj->clear(); });
    _nodes.resize(nnodes);
    _union_find.reset(nnodes); // reset union-find structure

    _degree_to_principle.clear();
    const ROWTYPE base = ai[0];
    for (COLTYPE i = 0; i < nnodes; i++)
    {
      _nodes[i].adjacent_variables.reserve(ai[i + 1] - ai[i]);
      for (ROWTYPE j_idx = ai[i] - base; j_idx < ai[i + 1] - base; j_idx++)
      {
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
                                          COLTYPE *perm)
  {
    initialize(nnodes, ai, aj);
  }

  template <typename COLTYPE>
  bool QuotientGraph<COLTYPE>::isDistinguishable(const COLTYPE i,
                                                 const COLTYPE j) const
  {
    // diff in quotient graph
    if (_nodes[i].adjacent_variables.size() !=
        _nodes[j].adjacent_variables.size())
      return true;
    if (_nodes[i].adjacent_elements.size() != _nodes[j].adjacent_elements.size())
      return true;

    // check if they have the same adjacent variables
    size_t k = 0, l = 0;
    while (k < _nodes[i].adjacent_variables.size() &&
           l < _nodes[j].adjacent_variables.size())
    {
      if (_nodes[i].adjacent_variables[k] != _nodes[j].adjacent_variables[l])
      {
        if (_nodes[i].adjacent_variables[k] == j)
          k++;
        else if (_nodes[j].adjacent_variables[l] == i)
          l++;
        else
          return true; // different neighbours
      }
      else
      {
        k++;
        l++;
      }
    }
    if (k < _nodes[i].adjacent_variables.size())
    {
      if (_nodes[i].adjacent_variables[k] == j)
        k++;
      else
        return true; // different neighbours
    }
    if (l < _nodes[j].adjacent_variables.size())
    {
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
           l < _nodes[j].adjacent_elements.size())
    {
      if (_nodes[i].adjacent_elements[k] != _nodes[j].adjacent_elements[l])
      {
        return true;
      }
      else
      {
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
                                          std::vector<COLTYPE> &temp)
  {
    assert(i == _union_find.Find(i));
    principleVector(_nodes[i].simple_variables, temp);
    principleVector(_nodes[i].adjacent_variables, temp);
    principleVector(_nodes[i].adjacent_elements, temp);
  }

  template <typename COLTYPE>
  void QuotientGraph<COLTYPE>::getFillins(const COLTYPE i,
                                          std::vector<COLTYPE> &temp1,
                                          std::vector<COLTYPE> &temp2)
  {
    // TODO: check if really needed
    updateNode(i, temp2);

    __vectors.clear();
    __vectors.push_back(&_nodes[i].adjacent_variables);

    for (auto j : _nodes[i].adjacent_elements)
    {
      updateNode(j, temp2);
      __vectors.push_back(&_nodes[j].adjacent_variables);
    }

    temp1.clear();
    mergeKVectors(__vectors, temp1);
    vectorSubtract(temp1, _nodes[i].simple_variables);
  }

  template <typename COLTYPE>
  void QuotientGraph<COLTYPE>::merge(const COLTYPE i, const COLTYPE j)
  {
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
    _nodes[j].simple_variables.shrink_to_fit();
    _nodes[j].adjacent_elements.clear();
    _nodes[j].adjacent_elements.shrink_to_fit();
    _nodes[j].adjacent_variables.clear();
    _nodes[j].adjacent_variables.shrink_to_fit();
  }

  template <typename COLTYPE>
  void QuotientGraph<COLTYPE>::massElimination(const COLTYPE p)
  {
    auto temp1 = _pool.acquire();
    auto temp2 = _pool.acquire();
    getFillins(p, *temp1, *temp2);
    auto &fill_ins = *temp1;
    for (auto i : fill_ins)
    {
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
      const std::vector<COLTYPE> &fillins)
  {
    const COLTYPE invalid = std::numeric_limits<COLTYPE>::max();
    auto temp1 = _pool.acquire();
    auto temp2 = _pool.acquire();
    std::map<COLTYPE, std::vector<COLTYPE>> supervariables;
    for (size_t i = 0; i < fillins.size(); ++i)
    {
      (*temp1).push_back(fillins[i]);
      (*temp2).push_back(hash(fillins[i]));
    }
    std::sort((*temp1).begin(), (*temp1).end(),
              [p = temp2.get()](const COLTYPE a, const COLTYPE b)
              {
                return (*p)[a] < (*p)[b];
              });
    for (size_t i = 0; i < fillins.size(); ++i)
    {
      (*temp2)[i] = hash((*temp1)[i]);
    }
    COLTYPE degree = 0;
    for (size_t i, j = 0; i < (*temp1).size();)
    {
      if ((*temp1)[i] == invalid)
      {
        continue; // already merged
      }
      degree = (*temp2)[i];
      j = i + 1;
      while (j < (*temp1).size() && (*temp2)[j] == degree)
      {
        if ((*temp1)[j] == invalid)
        {
          j++;
          continue; // already merged
        }
        if (isDistinguishable((*temp1)[i], (*temp1)[j]))
        {
          j++;
          continue; // cannot merge
        }
        merge((*temp1)[i], (*temp1)[j]);
        (*temp1)[j] = invalid;
      }
      i = j;
    }
  }

  template <typename COLTYPE>
  COLTYPE QuotientGraph<COLTYPE>::getExternalDegree(const COLTYPE i)
  {
    auto temp1 = _pool.acquire();
    auto temp2 = _pool.acquire();
    getFillins(i, *temp1, *temp2);
    return static_cast<COLTYPE>(temp2->size());
  }

  template <typename COLTYPE>
  COLTYPE QuotientGraph<COLTYPE>::hash(const COLTYPE i) const
  {
    COLTYPE adjacent_nodes = _nodes[i].adjacent_variables.size() +
                             _nodes[i].adjacent_elements.size();
    return adjacent_nodes % (static_cast<COLTYPE>(_nodes.size()) - 1) + 1;
  }

  template <typename COLTYPE>
  void QuotientGraph<COLTYPE>::principleVector(std::vector<COLTYPE> &vec,
                                               std::vector<COLTYPE> &temp)
  {
    temp.clear();
    for (auto i : vec)
    {
      temp.push_back(_union_find.Find(i));
    }
    std::sort(temp.begin(), temp.end());
    vec.clear();
    for (auto i : temp)
    {
      if (vec.empty() || vec.back() != i)
      {
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