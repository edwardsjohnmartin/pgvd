#ifndef __LINE_TRANSFORM_H__
#define __LINE_TRANSFORM_H__

#define M_PI 3.14159

inline void floatn_swap(floatn* a, floatn* b) {
  floatn temp = *a;
  *a = *b;
  *b = temp;
}

// Matrix:
//    0 1 2
//    3 4 5
//    6 7 8

// Identity matrix
inline float* I(float* m) {
  m[0] = m[4] = m[8] = 1;
  m[1] = m[2] = m[3] = m[5] = m[6] = m[7] = 0;
  return m;
}

inline float* rotation(float theta, float* m) {
  I(m);
  m[0] = cos(theta);
  m[1] = -sin(theta);
  m[3] = sin(theta);
  m[4] = cos(theta);
  return m;
}

inline float* translation(float x, float y, float* m) {
  I(m);
  m[2] = x;
  m[5] = y;
  return m;
}

inline float* reflect0(float* m) {
  I(m);
  m[4] = -1;
  return m;
}

inline float* reflect45(float* m) {
  I(m);
  m[0] = 0;
  m[1] = 1;
  m[3] = 1;
  m[4] = 0;
  return m;
}

inline float* mult(float* m, float* n, const int store) {
  float result[9];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      result[i*3+j] = m[i*3]*n[j] + m[i*3+1]*n[3+j] + m[i*3+2]*n[6+j];
    }
  }

  if (store == 0) {
    for (int i = 0; i < 9; ++i) {
      m[i] = result[i];
    }
    return m;
  } else {
    for (int i = 0; i < 9; ++i) {
      n[i] = result[i];
    }
    return n;
  }
}

inline floatn apply_point(floatn* p, const float* m) {
  floatn q;
  q.x = p->x * m[0] + p->y * m[1] + m[2];
  q.y = p->x * m[3] + p->y * m[4] + m[5];
  *p = q;
  return *p;
}

inline floatn apply_vector(floatn* p, const float* m) {
  floatn q;
  q.x = p->x * m[0] + p->y * m[1];
  q.y = p->x * m[3] + p->y * m[4];
  *p = q;
  return *p;
}

typedef struct LineTransform {
//
//  void reflect(const floatn u) {
//    float temp[9];
//    const float gamma = atan2(u.y, u.x);
//    if (gamma > M_PI/4) {
//      mult(rotation(-M_PI/2, temp), _m, 1);
//      mult(_m_inv, rotation(M_PI/2, temp), 0);
//      _swap = true;
//    } else if (gamma < -M_PI/4) {
//      mult(reflect45(temp), mult(rotation(M_PI/2, temp), _m, 1), 1);
//      mult(_m_inv, mult(rotation(-M_PI/2, temp), reflect45(temp), 0), 0);
//      _swap = true;
//    } else if (gamma < 0) {
//      mult(reflect0(temp), _m, 1);
//      mult(_m_inv, reflect0(temp), 0);
//      _swap = true;
//    }
//  }
//
//  void apply(floatn* q0, floatn* r0, floatn* p0,
//             floatn* v, floatn* w, floatn* u) const {
//    apply(q0, r0, p0, v, w, u, _m);
//  }
//
//  void applyPoint(floatn* p) const {
//    apply_point(p, _m);
//  }
//
//  void revert(floatn* q0, floatn* r0, floatn* p0,
//            floatn* v, floatn* w, floatn* u) const {
//    apply(q0, r0, p0, v, w, u, _m_inv);
//  }
//
//  void revertPoint(floatn* p) const {
//    apply_point(p, _m_inv);
//  }
//
//  void revertVector(floatn* v) const {
//    *v = apply_vector(v, _m_inv);
//  }
//
//  // void print() const {
//  //   for (int i = 0; i < 9; ++i) {
//  //     std::cout << _m[i] << " ";
//  //     if (i%3 == 2) {
//  //       std::cout << std::endl;
//  //     }
//  //   }
//  // }
//
//
  float _m[9];
  float _m_inv[9];
  bool _swap;
} LineTransform;

inline void InitLineTransform(LineTransform *LT, const floatn* u, const floatn* origin) {
  float ptheta = atan2(u->y, u->x);
  while (ptheta < 0) ptheta += 2 * M_PI;

  float temp[9];

  translation(-origin->x, -origin->y, LT->_m);
  translation(origin->x, origin->y, LT->_m_inv);

  const float rotate = floor(ptheta / (M_PI/2)) * (M_PI/2);
  if (rotate > 0) {
    mult(rotation(-rotate, temp), LT->_m, 1);
    mult(LT->_m_inv, rotation(rotate, temp), 0);
  }

	LT->_swap = (((int)floor(ptheta / (M_PI/4))) % 2) == 1;
  if (LT->_swap) {
    mult(reflect45(temp), LT->_m, 1);
    mult(LT->_m_inv, reflect45(temp), 0);
  }
}

__inline void applyToLineTransform(LineTransform* LT, floatn* q0, floatn* r0, floatn* p0,
		floatn* v, floatn* w, floatn* u) {
	apply_point(q0, LT->_m);
	apply_point(r0, LT->_m);
	apply_point(p0, LT->_m);
	apply_vector(v, LT->_m);
	apply_vector(w, LT->_m);
	apply_vector(u, LT->_m);
	
	if (LT->_swap) {
	  floatn_swap(q0, r0);
	  floatn_swap(v, w);
	}
}

__inline void revertLineTransform(LineTransform* LT, floatn* q0, floatn* r0, floatn* p0,
	floatn* v, floatn* w, floatn* u) {
	apply_point(q0, LT->_m_inv);
	apply_point(r0, LT->_m_inv);
	apply_point(p0, LT->_m_inv);
	apply_vector(v, LT->_m_inv);
	apply_vector(w, LT->_m_inv);
	apply_vector(u, LT->_m_inv);

	if (LT->_swap) {
		floatn_swap(q0, r0);
		floatn_swap(v, w);
	}
}

__inline void revertLineTransformPoint(const LineTransform *LT, floatn* p) {
	apply_point(p, LT->_m_inv);
};
__inline void revertLineTransformVector(const LineTransform *LT,  floatn* v) {
	*v = apply_vector(v, LT->_m_inv);
};

#endif
