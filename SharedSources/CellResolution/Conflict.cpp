#ifdef __OPENCL_VERSION__
#include "./SharedSources/CellResolution/Conflict.h"
#include "./SharedSources/CellResolution/LineTransform.h"
#elif defined __cplusplus
#include "./Conflict.h"
#include "./LineTransform.h"
#endif

//------------------------------------------------------------
//------------------------------------------------------------
// Contents of the H file
//------------------------------------------------------------
//------------------------------------------------------------

inline void floatn_array_add(const floatn* value, floatn_array* array) {
    if (array->array) {
        array->array[array->i++] = *value;
    }
    else {
        array->i++;
    }
}

//------------------------------------------------------------
// Vector functions
//------------------------------------------------------------

// The norm (length) of the cross product
inline float ncross(const floatn* u, const floatn* v) {
    return u->x * v->y - u->y * v->x;
}

//------------------------------------------------------------
// LineSegment
//------------------------------------------------------------
typedef struct LineSegment {
    floatn p0, p1;
} LineSegment;
inline LineSegment make_segment_from_points(const floatn* p0, const floatn* p1) {
    LineSegment s = { *p0, *p1 };
    return s;
}
inline void reverse_segment(LineSegment* s) {
    floatn temp = s->p0;
    s->p0 = s->p1;
    s->p1 = temp;
}
#ifdef __cplusplus
inline std::ostream& operator<<(std::ostream& out, const LineSegment& s) {
    out << "(" << s.p0 << ") (" << s.p1 << ")";
    return out;
}
#endif

//------------------------------------------------------------
// LineSegmentPair
//------------------------------------------------------------
typedef struct LineSegmentPair {
    LineSegment s0, s1;
} LineSegmentPair;
inline LineSegmentPair make_line_segment_pair(const LineSegment* s0, const LineSegment* s1) {
    LineSegmentPair p = { *s0, *s1 };
    return p;
}

//------------------------------------------------------------
// Bounding box
//------------------------------------------------------------
typedef struct BB {
    floatn o; // origin
    float w, h; // width and height
    bool empty;
} BB;
#ifdef __cplusplus
inline std::ostream& operator<<(std::ostream& out, const BB& bb) {
    out << "(" << bb.o << ") (" << (bb.o + make_floatn(bb.w, bb.h)) << ")";
    return out;
}
#endif
//inline BB make_bb() {
//  BB bb = { make_floatn(0,0), 0, 0, true };
//  return bb;
//}
inline BB make_bb(const floatn* origin, const float w) {
    BB bb = { *origin, w, w, false };
    return bb;
}
inline void add_to_bb(const floatn* p, BB* bb) {
    if (bb->empty) {
        bb->empty = false;
        bb->o = *p;
        bb->w = 0;
        bb->h = 0;
    }
    else {
        if (p->x < bb->o.x) {
            bb->w += (bb->o.x - p->x);
            bb->o.x = p->x;
        }
        else if (p->x > bb->o.x + bb->w) {
            bb->w = p->x - bb->o.x;
        }
        if (p->y < bb->o.y) {
            bb->h += (bb->o.y - p->y);
            bb->o.y = p->y;
        }
        else if (p->y > bb->o.y + bb->h) {
            bb->h = p->y - bb->o.y;
        }
    }
}

//------------------------------------------------------------
// Auxiliary function declarations
//------------------------------------------------------------

int get_line_segment_pairs(
    const LineSegment* s0_, const LineSegment* s1_,
    LineSegmentPair pairs[],
    const floatn* origin, const float width);

void line_box_intersection(
    const floatn* p0, const floatn* v,
    const floatn* o, const float w, const float h,
    int* num_intersections, float* t0, float* t1);

bool inside_square(const floatn* p, const floatn* o, const float w);

LineSegment clip_segment(
    const LineSegment* s_, const BB* bb, bool* valid);

void orientLines(floatn* q0, floatn* v,
    floatn* r0, floatn* w);

//------------------------------------------------------------
// Auxiliary inline function definitions
//------------------------------------------------------------


//------------------------------------------------------------
// Functions from the paper
//------------------------------------------------------------

float a_f(const bool opposite, const float s,
    const floatn* p0, const floatn* q0, const floatn* r0,
    const floatn* u, const floatn* v, const floatn* w);

float alpha_f(const bool opposite,
    const floatn* p0, const floatn* q0, const floatn* r0,
    const floatn* u, const floatn* v, const floatn* w);

float beta_f(const bool opposite,
    const floatn* p0, const floatn* q0, const floatn* r0,
    const floatn* u, const floatn* v, const floatn* w);


//------------------------------------------------------------
//------------------------------------------------------------
// End H file
//------------------------------------------------------------
//------------------------------------------------------------


//------------------------------------------------------------
// Ray
//------------------------------------------------------------
typedef struct Ray {
    floatn p0, v;
} Ray;
inline Ray make_ray_from_segment(const LineSegment* s) {
    Ray r = { s->p0, s->p1 - s->p0 };
    return r;
}
inline Ray make_ray_from_point_and_vec(const floatn* p0, const floatn* v) {
    Ray r = { *p0, *v };
    return r;
}
inline LineSegment make_segment_from_ray(const Ray* r) {
    LineSegment s = { r->p0, r->p0 + r->v };
    return s;
}
inline floatn get_ray_point(const Ray* r, const float t) {
    return r->p0 + r->v * t;
}
inline void reverse_ray(Ray* r) {
    floatn p0 = r->p0 + r->v;
    r->v = r->v*(-1);
    r->p0 = p0;
}

//------------------------------------------------------------
// RayPair
//------------------------------------------------------------
typedef struct RayPair {
    Ray r0, r1;
} RayPair;
inline RayPair make_ray_pair(const Ray* r0, const Ray* r1) {
    RayPair p = { *r0, *r1 };
    return p;
}

//------------------------------------------------------------
// Local auxiliary function declarations
//------------------------------------------------------------

void clip_v_half(LineSegment* a, LineSegment* b,
    const bool a_half, const bool b_half);
inline bool inside_box(const floatn* p, const BB* bb);
inline bool inside_rect(
    const floatn* p, const floatn* o, const float w, const float h);
inline bool line_line_intersection(const floatn* q0, const floatn* v,
    const floatn* r0, const floatn* w,
    floatn* p, float* t, float* f);
inline float line_point_intersection(const floatn* q0, const floatn* v,
    const floatn* p);
inline float ray_point_intersection(const Ray* r, const floatn* p);
inline float ray_axis_intersection(const Ray* r, const float f, const int axis);
inline void ray_box_intersection(
    const Ray* r, const BB* bb,
    int* num_intersections, float* t0, float* t1);


//------------------------------------------------------------
// Auxiliary function definitions
//------------------------------------------------------------

// Given a ray r, returns the intersection of the ray with the
// line segment s. Points along the negative direction of the
// ray are not included. Returns false if the resulting
// line segment is empty;
inline bool clip_ray(
    const Ray* r, const LineSegment* s,
    LineSegment* result) {
    float t0 = ray_point_intersection(r, &s->p0);
    float t1 = ray_point_intersection(r, &s->p1);
    // Order the two points
    if (t1 < t0) {
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }
    const floatn p0 = get_ray_point(r, t0);
    const floatn p1 = get_ray_point(r, t1);

    if (t1 < 0) return false;
    if (t0 < 0) {
        *result = make_segment_from_points(&r->p0, &p1);
    }
    else {
        *result = make_segment_from_points(&p0, &p1);
    }
    return true;
}

// Returns a line segment that is a subset of s clipped to the given box.
LineSegment clip_segment(
    const LineSegment* s_, const BB* bb, bool* valid) {
    LineSegment s = *s_;

    Ray r = make_ray_from_segment(&s);
    int num_intersections;
    float t0, t1;
    ray_box_intersection(&r, bb, &num_intersections, &t0, &t1);

    *valid = false;
    if (t0 > 0 - EPSILON && t0 < 1 + EPSILON) {
        s.p0 = get_ray_point(&r, t0);
        *valid = true;
    }
    if (t1 > 0 - EPSILON && t1 < 1 + EPSILON) {
        s.p1 = get_ray_point(&r, t1);
        *valid = true;
    }
    if (!*valid) {
        *valid = (t0 < 0 && t1 > 0);
    }
    return s;
}

// Returns true if point p is inside the box.
// The box is considered closed, so points on the boundary are inside.
inline bool inside_box(const floatn* p, const BB* bb) {
    return
        p->x >= bb->o.x - EPSILON && // left
        p->x <= bb->o.x + bb->w + EPSILON && // right
        p->y >= bb->o.y - EPSILON && // bottom
        p->y <= bb->o.y + bb->h + EPSILON; // top
}

// Given a box origin, width and height,
// returns true if point p is inside the box.
// The box is considered closed, so points on the boundary are inside.
inline bool inside_rect(
    const floatn* p, const floatn* o, const float w, const float h) {
    return
        p->x >= o->x - EPSILON && // left
        p->x <= o->x + w + EPSILON && // right
        p->y >= o->y - EPSILON && // bottom
        p->y <= o->y + h + EPSILON; // top
}

// Given a box origin and width, returns true if point p is inside the box.
// The box is considered closed, to points on the boundary are inside.
bool inside_square(const floatn* p, const floatn* o, const float w) {
    return inside_rect(p, o, w, w);
}

// Given two lines in parametric form, find the intersection point. The
// intersection is given in terms of parameters t and f of the two lines, as
// well as the point p itself.
// Returns false if the lines are parallel.
inline bool line_line_intersection(const floatn* q0, const floatn* v,
    const floatn* r0, const floatn* w,
    floatn* p, float* t, float* f) {
    // Find where q and r intersect
    const float den = v->x * w->y - v->y * w->x;
    if (fabs(den) < EPSILON) return false;

    // Not parallel
    *t = (w->y * (r0->x - q0->x) + w->x * (q0->y - r0->y)) / den;
    *f = (v->y * (q0->x - r0->x) + v->x * (r0->y - q0->y)) / (-den);
    *p = (*q0) + (*v) * (*t);

    return true;
}

// Given a line in parametric form, find the intersection point with a point.
// Intersection test is NOT made, but it is assumed that the line and point
// do indeed intersect.
inline float line_point_intersection(const floatn* q0, const floatn* v,
    const floatn* p) {
    if (fabs(v->x) > fabs(v->y)) {
        return (p->x - q0->x) / v->x;
    }
    return (p->y - q0->y) / v->y;
}

// Given a line in parametric form, find the intersection point with a point.
// Intersection test is NOT made, but it is assumed that the line and point
// do indeed intersect.
inline float ray_point_intersection(const Ray* r, const floatn* p) {
    if (fabs(r->v.x) > fabs(r->v.y)) {
        return (p->x - r->p0.x) / r->v.x;
    }
    return (p->y - r->p0.y) / r->v.y;
}

// Given a line in parametric form, find the intersection point with an
// axis-aligned line.
// axis = 0: aligned with x axis
// axis = 1: aligned with y axis
// axis = 2: aligned with z axis
inline float ray_axis_intersection(
    const Ray* r, const float f, const int axis) {
    if (axis == 1) {
        return (f - r->p0.x) / r->v.x;
    }
    return (f - r->p0.y) / r->v.y;

}

// Given a line in parametric form p = tv, find the <=two intersection points
// with a box at origin o with width w and height h. t values returned are
// sorted in ascending order.
void line_box_intersection(
    const floatn* p0, const floatn* v,
    const floatn* o, const float w, const float h,
    int* num_intersections, float* t0, float* t1) {
    if (fabs(v->x) < EPSILON) {
        // vertical line
        *t0 = (o->y - p0->y) / v->y;
        *t1 = ((o->y + h) - p0->y) / v->y;
        if (p0->x >= o->x - EPSILON && // left
            p0->x <= o->x + w + EPSILON) {
            *num_intersections = 2;
        }
        else {
            *num_intersections = 0;
        }
    }
    else if (fabs(v->y) < EPSILON) {
        // horizontal line
        *t0 = (o->x - p0->x) / v->x;
        *t1 = ((o->x + w) - p0->x) / v->x;
        if (p0->y >= o->y - EPSILON && // left
            p0->y <= o->y + h + EPSILON) {
            *num_intersections = 2;
        }
        else {
            *num_intersections = 0;
        }
    }
    else {
        const float tleft = (o->x - p0->x) / v->x;
        const float tright = ((o->x + w) - p0->x) / v->x;
        const float tbottom = (o->y - p0->y) / v->y;
        const float ttop = ((o->y + h) - p0->y) / v->y;
        if (v->x > 0) {
            // Vector traveling left to right
            *t0 = fmax(tleft, fmin(ttop, tbottom));
            *t1 = fmin(tright, fmax(ttop, tbottom));
        }
        else {
            // Vector traveling right to left
            *t0 = fmax(tright, fmin(ttop, tbottom));
            *t1 = fmin(tleft, fmax(ttop, tbottom));
        }
        floatn q = (*p0) + (*v) * (*t0);
        if (inside_rect(&q, o, w, h)) {
            *num_intersections = 2;
        }
        else {
            *num_intersections = 0;
        }
    }
    if (*num_intersections == 2 && *t0 > *t1) {
        float temp = *t0;
        *t0 = *t1;
        *t1 = temp;
    }
}

inline void ray_box_intersection(
    const Ray* r, const BB* bb,
    int* num_intersections, float* t0, float* t1) {
    line_box_intersection(&r->p0, &r->v, &bb->o, bb->w, bb->h,
        num_intersections, t0, t1);
}

// Given a "v" (two line segments emanating from their intersection point),
// clips the segments based on which segment is a "half" segment. Half
// segments are defined as a segment which does not contain the intersection
// point.
void clip_v_half(LineSegment* a, LineSegment* b,
    const bool a_half, const bool b_half) {
    floatn origin = make_floatn(0, 0);
    BB bb = make_bb(&origin, 0);
    add_to_bb(&a->p1, &bb);
    add_to_bb(&b->p1, &bb);
    if (a_half) {
        add_to_bb(&a->p0, &bb);
    }
    else {
        add_to_bb(&b->p0, &bb);
    }
    bool a_valid, b_valid;
    *a = clip_segment(a, &bb, &a_valid);
    *b = clip_segment(b, &bb, &b_valid);
}

void flipRay(floatn* q0, floatn* v) {
    floatn temp = *q0;
    *q0 = (*q0) + (*v);
    *v = temp - (*q0);
}

void orientLines(floatn* q0, floatn* v,
    floatn* r0, floatn* w) {
    // Find where q and r intersect
    const float den = v->x * w->y - v->y * w->x;
    if (fabs(den) > EPSILON) {
        // Not parallel
        const float t_origin =
            (w->y * (r0->x - q0->x) + w->x * (q0->y - r0->y)) / den;
        const float f_origin =
            (v->y * (q0->x - r0->x) + v->x * (r0->y - q0->y)) / (-den);

        // Want angle of w less than angle of v.
        const float3 c =
            cross(make_float3(v->x, v->y, 0), make_float3(w->x, w->y, 0));
        if (c.z > 0) {
            // swap(*q0, *r0);
            // swap(*v, *w);
            floatn_swap(q0, r0);
            floatn_swap(v, w);
        }
    }
    else {
        // parallel
        if (dot(*v, *w) < 0) {
            flipRay(q0, v);
        }
    }
}


// Given two line segments, makes potentially four segment pairs that would
// need to be resolved.  Suppose we have a red and a blue line.
//   case 1: the lines intersect and the intersection point is in the
//      bounding box. Then four pairs are returned, one for each red/blue pair.
//      The segments returned are in "v" form, or the segments clipped at
//      the intersection point.
//   case 2: the lines intersect/don't intersect but the intersection point
//      is/would be outside the bounding box. One pair is returned: the
//      intersection of the two lines with the bounding box.
//   case 3: the lines don't intersect and the intersection point would be
//      inside the bounding box. Two pairs are returned: suppose the red
//      segment is the "half" segment, or the one that doesn't reach the
//      intersection point. Then the first pair is the red segment with the
//      part of the blue segment on the positive side of the would-be
//      intersection. The second pair is the red segment with the part of the
//      blue segment on the negative side of intersection.
// Return value is the number of pairs.
int get_line_segment_pairs(
    const LineSegment* s0_, const LineSegment* s1_,
    LineSegmentPair pairs[],
    const floatn* origin, const float width) {

    Ray r0 = make_ray_from_segment(s0_);
    Ray r1 = make_ray_from_segment(s1_);
    if (dot(r0.v, r1.v) < 0) {
        reverse_ray(&r0);
    }
    LineSegment s0 = make_segment_from_ray(&r0);
    LineSegment s1 = make_segment_from_ray(&r1);

    BB bb = make_bb(origin, width);

    const floatn v0 = r0.v;
    const floatn v0n = v0*-1;
    const floatn v1 = r1.v;
    const floatn v1n = v1*-1;

    // p is the point at which the two lines intersect.
    floatn p;
    float t0, t1;
    const bool parallel =
        !line_line_intersection(&r0.p0, &r0.v, &r1.p0, &r1.v, &p, &t0, &t1);
    // half_intersection means that if one of the segments was
    // extended then the line segments would intersect. a/b_half
    // means that the line segment does not contain the intersection
    // point.
    const bool a_half = (t0 < 0 || t0 > 1);
    const bool b_half = (t1 < 0 || t1 > 1);
    const bool half_intersection = (a_half != b_half);
    const bool full_intersection = (!a_half && !b_half);

    Ray a = make_ray_from_point_and_vec(&p, &v0);
    Ray b = make_ray_from_point_and_vec(&p, &v1);
    // In the negative directions
    Ray an = make_ray_from_point_and_vec(&p, &v0n);
    Ray bn = make_ray_from_point_and_vec(&p, &v1n);
    int n;
    int i = 0;
    if (parallel) {
        // Lines are parallel
        pairs[i++] = make_line_segment_pair(&s0, &s1);
    }
    else {
        LineSegment ac, bc;
        if (clip_ray(&a, &s0, &ac) && clip_ray(&b, &s1, &bc)) {
            if (half_intersection) {
                clip_v_half(&ac, &bc, a_half, b_half);
            }
            else if (inside_box(&ac.p1, &bb)) {
                Ray bc_ray = make_ray_from_segment(&bc);
                float t0 = ray_axis_intersection(&bc_ray, ac.p1.x, 1);
                float t1 = ray_axis_intersection(&bc_ray, ac.p1.y, 0);
                if (t0 > 0) {
                    bc.p1 = get_ray_point(&bc_ray, t0);
                }
                else if (t1 > 0) {
                    bc.p1 = get_ray_point(&bc_ray, t1);
                }
            }
            else if (inside_box(&bc.p1, &bb)) {
                Ray ac_ray = make_ray_from_segment(&ac);
                float t0 = ray_axis_intersection(&ac_ray, bc.p1.x, 1);
                float t1 = ray_axis_intersection(&ac_ray, bc.p1.y, 0);
                if (t0 > 0) {
                    ac.p1 = get_ray_point(&ac_ray, t0);
                }
                else if (t1 > 0) {
                    ac.p1 = get_ray_point(&ac_ray, t1);
                }
            }
            pairs[i++] = make_line_segment_pair(&ac, &bc);
        }
        if (clip_ray(&a, &s0, &ac) && clip_ray(&bn, &s1, &bc)) {
            if (half_intersection) {
                clip_v_half(&ac, &bc, a_half, b_half);
            }
            else if (inside_box(&ac.p1, &bb)) {
                Ray bc_ray = make_ray_from_segment(&bc);
                float t0 = ray_axis_intersection(&bc_ray, ac.p1.x, 1);
                float t1 = ray_axis_intersection(&bc_ray, ac.p1.y, 0);
                if (t0 > 0) {
                    bc.p1 = get_ray_point(&bc_ray, t0);
                }
                else if (t1 > 0) {
                    bc.p1 = get_ray_point(&bc_ray, t1);
                }
            }
            else if (inside_box(&bc.p1, &bb)) {
                Ray ac_ray = make_ray_from_segment(&ac);
                float t0 = ray_axis_intersection(&ac_ray, bc.p1.x, 1);
                float t1 = ray_axis_intersection(&ac_ray, bc.p1.y, 0);
                if (t0 > 0) {
                    ac.p1 = get_ray_point(&ac_ray, t0);
                }
                else if (t1 > 0) {
                    ac.p1 = get_ray_point(&ac_ray, t1);
                }
            }
            pairs[i++] = make_line_segment_pair(&ac, &bc);
        }
        if (clip_ray(&an, &s0, &ac) && clip_ray(&b, &s1, &bc)) {
            if (half_intersection) {
                clip_v_half(&ac, &bc, a_half, b_half);
            }
            else if (inside_box(&ac.p1, &bb)) {
                Ray bc_ray = make_ray_from_segment(&bc);
                float t0 = ray_axis_intersection(&bc_ray, ac.p1.x, 1);
                float t1 = ray_axis_intersection(&bc_ray, ac.p1.y, 0);
                if (t0 > 0) {
                    bc.p1 = get_ray_point(&bc_ray, t0);
                }
                else if (t1 > 0) {
                    bc.p1 = get_ray_point(&bc_ray, t1);
                }
            }
            else if (inside_box(&bc.p1, &bb)) {
                Ray ac_ray = make_ray_from_segment(&ac);
                float t0 = ray_axis_intersection(&ac_ray, bc.p1.x, 1);
                float t1 = ray_axis_intersection(&ac_ray, bc.p1.y, 0);
                if (t0 > 0) {
                    ac.p1 = get_ray_point(&ac_ray, t0);
                }
                else if (t1 > 0) {
                    ac.p1 = get_ray_point(&ac_ray, t1);
                }
            }
            pairs[i++] = make_line_segment_pair(&ac, &bc);
        }
        if (clip_ray(&an, &s0, &ac) && clip_ray(&bn, &s1, &bc)) {
            if (half_intersection) {
                clip_v_half(&ac, &bc, a_half, b_half);
            }
            else if (inside_box(&ac.p1, &bb)) {
                Ray bc_ray = make_ray_from_segment(&bc);
                float t0 = ray_axis_intersection(&bc_ray, ac.p1.x, 1);
                float t1 = ray_axis_intersection(&bc_ray, ac.p1.y, 0);
                if (t0 > 0) {
                    bc.p1 = get_ray_point(&bc_ray, t0);
                }
                else if (t1 > 0) {
                    bc.p1 = get_ray_point(&bc_ray, t1);
                }
            }
            else if (inside_box(&bc.p1, &bb)) {
                Ray ac_ray = make_ray_from_segment(&ac);
                float t0 = ray_axis_intersection(&ac_ray, bc.p1.x, 1);
                float t1 = ray_axis_intersection(&ac_ray, bc.p1.y, 0);
                if (t0 > 0) {
                    ac.p1 = get_ray_point(&ac_ray, t0);
                }
                else if (t1 > 0) {
                    ac.p1 = get_ray_point(&ac_ray, t1);
                }
            }
            pairs[i++] = make_line_segment_pair(&ac, &bc);
        }
    }

    return i;
}

//------------------------------------------------------------
// Functions from the paper
//------------------------------------------------------------

float a_f(const bool opposite, const float s,
    const floatn* p0, const floatn* q0, const floatn* r0,
    const floatn* u, const floatn* v, const floatn* w) {
    if (opposite) {
        const float num = (p0->x + s*u->x - q0->x)*ncross(w, v) +
            v->x*(ncross(r0, w) + ncross(w, q0));
        const float den = v->x*(w->x + w->y);
        return num / den;
    }
    else {
        return s*u->x*ncross(w, v) / (v->x*w->x) +
            q0->y - r0->y + v->y*(p0->x - q0->x) / v->x - w->y*(p0->x - r0->x) / w->x;
    }
}

float alpha_f(const bool opposite,
    const floatn* p0, const floatn* q0, const floatn* r0,
    const floatn* u, const floatn* v, const floatn* w) {
    if (opposite) {
        return 1 + ncross(w, v) / (v->x*(w->x + w->y));
    }
    else {
        return 1 + ncross(w, v) / (v->x*w->x);
    }
}

float beta_f(const bool opposite,
    const floatn* p0, const floatn* q0, const floatn* r0,
    const floatn* u, const floatn* v, const floatn* w) {
    if (opposite) {
        const float num = (p0->x - q0->x)*ncross(w, v) +
            v->x*(ncross(r0, w) + ncross(w, q0));
        const float den = u->x*v->x*(w->x + w->y);
        return num / den;
    }
    else {
        return (q0->y - r0->y + v->y*(p0->x - q0->x) / v->x - w->y*(p0->x - r0->x) / w->x) / u->x;
    }
}

void add_sample(
    const float s,
    const floatn p_origin, const floatn u,
    const floatn p_origin_t, const floatn u_t,
    const floatn q0_t, const floatn v_t,
    const floatn r0_t, const floatn w_t,
    const bool opposite,
    const floatn origin0, const int width0,
    const LineTransform T,
    floatn_array* samples) {

    const floatn p = p_origin + u*s;
    // cout << "s = " << s << " p = " << p << endl;

    if (!inside_square(&p, &origin0, width0)) return;

    floatn_array_add(&p, samples);
}

// origin_ is the clipped origin, along with width and height.
// origin0 and width0 are the original cell bounds.
void sample_v_conflict(
    floatn q0, floatn v,
    floatn r0, floatn w,
    floatn origin_, int width, int height,
    floatn origin0, int width0,
    floatn_array* samples) {
    // p = p0 + su
    // q = q0 + tv
    // r = r0 + fw

    // Handle starting with an intersection
#ifdef __OPENCL_VERSION__
    if ((q0.x == r0.x) && (q0.y == r0.y)) {
        // HACK. Just moving the points one unit along their vectors.
        q0 = q0 + v / length(v);
        r0 = r0 + w / length(w);
    }
#else
    if (q0 == r0) {
        // HACK. Just moving the points one unit along their vectors.
        q0 = q0 + v / length(v);
        r0 = r0 + w / length(w);
    }
#endif
    orientLines(&q0, &v, &r0, &w);

    floatn p_origin = (q0 + r0) / 2;

    float qtheta = atan2(v.y, v.x);
    float rtheta = atan2(w.y, w.x);
    if (qtheta < rtheta) qtheta += 2 * M_PI;
    float ptheta = (qtheta + rtheta) / 2;
    floatn u = make_floatn(cos(ptheta), sin(ptheta));

    // Find the s values where u enters and exits the bounding box
    float s0, sn;
    int num_intersections;
    line_box_intersection(&p_origin, &u, &origin_, width, height,
        &num_intersections, &s0, &sn);
    if (s0 < 0) {
        s0 = 0;
    }
    if (sn < 0) {
        sn = 0;
        printf("Error: sn < 0\n");
    }

    LineTransform T;
    InitLineTransform(&T, &u, &p_origin);
    applyToLineTransform(&T, &q0, &r0, &p_origin, &v, &w, &u);

    ptheta = atan2(u.y, u.x);
    qtheta = atan2(v.y, v.x);
    rtheta = atan2(w.y, w.x);

    // There are two types of sample lines. The first is "diagonal" or "opposite"
    // if the transformed w vector has w.y >= 0. The other is "adjacent" if
    // w.y < 0. We need the sample line to determine the transformation, so
    // we can determine sample line type by seeing if the v and w vectors span
    // an axis.
    const bool opposite = rtheta > -EPSILON;

    const float alpha = alpha_f(opposite, &p_origin, &q0, &r0, &u, &v, &w);
    const float beta = beta_f(opposite, &p_origin, &q0, &r0, &u, &v, &w);
    float a0 = a_f(opposite, s0, &p_origin, &q0, &r0, &u, &v, &w);

    const float s1 = s0 + a0 / (2 * u.x);
    const float k1_even = alpha*s0 + beta -
        (alpha*alpha*s0 + alpha*beta - alpha*s0) / (alpha - 1);
    const float k1_odd = alpha*s1 + beta -
        (alpha*alpha*s1 + alpha*beta - alpha*s1) / (alpha - 1);
    const float k2_even = (alpha*s0 + beta - s0) / (alpha - 1);
    const float k2_odd = (alpha*s1 + beta - s1) / (alpha - 1);

    const float log_alpha = log(alpha);
    const int max_even = (int)ceil(log((sn - k1_even) / k2_even) / log_alpha);
    const int max_odd = (int)ceil(log((sn - k1_odd) / k2_odd) / log_alpha);
    const int max_i = max_even + max_odd + 2;

    // After getting the s values we transform the bisector back to the original
    // frame. Get a copy of the transformed values first.
    const floatn q0_t = q0;
    const floatn r0_t = r0;
    const floatn p_origin_t = p_origin;
    const floatn v_t = v;
    const floatn w_t = w;
    const floatn u_t = u;
    revertLineTransform(&T, &q0, &r0, &p_origin, &v, &w, &u);

    if (max_i > 0) {
        add_sample(s0, p_origin, u, p_origin_t, u_t, q0_t, v_t, r0_t, w_t,
            opposite, origin0, width0, T, samples);
        if (max_i > 1) {
            add_sample(s1, p_origin, u, p_origin_t, u_t, q0_t, v_t, r0_t, w_t,
                opposite, origin0, width0, T, samples);
        }
    }
    for (int i = 2; i < max_i; ++i) {
        const float po = pow(alpha, i / 2);
        float s;
        if (i % 2 == 0) {
            s = k1_even + k2_even * po;
        }
        else {
            s = k1_odd + k2_odd * po;
        }
        add_sample(s, p_origin, u, p_origin_t, u_t, q0_t, v_t, r0_t, w_t,
            opposite, origin0, width0, T, samples);
    }
}

// The lines are given in "original" parameter space, where the endpoints
// are q0_, q1_, r0_, and r1_.
void sample_conflict_impl(
    const intn q0_int, const floatn v_,
    const intn r0_int, const floatn w_,
    intn origin__, int width, floatn_array* samples) {

    const floatn q0_ = convert_floatn(q0_int);
    const floatn q1_ = q0_ + v_;
    const floatn r0_ = convert_floatn(r0_int);
    const floatn r1_ = r0_ + w_;

    floatn origin_ = convert_floatn(origin__);
    floatn origin0 = origin_;
    int width0 = width;
    int height = width;

    LineSegment s0 = make_segment_from_points(&q0_, &q1_);
    LineSegment s1 = make_segment_from_points(&r0_, &r1_);

    LineSegmentPair pairs[4];
    const int num_pairs = get_line_segment_pairs(
        &s0, &s1, pairs, &origin_, width);

    const BB bb_ = make_bb(&origin_, width);

    for (int i = 0; i < num_pairs; ++i) {
        LineSegment* a = &pairs[i].s0;
        LineSegment* b = &pairs[i].s1;

        bool a_valid, b_valid;
        *a = clip_segment(a, &bb_, &a_valid);
        *b = clip_segment(b, &bb_, &b_valid);

        floatn origin = make_floatn(0, 0);
        BB small_bb = make_bb(&origin, 0);
        add_to_bb(&a->p0, &small_bb);
        add_to_bb(&a->p1, &small_bb);
        add_to_bb(&b->p0, &small_bb);
        add_to_bb(&b->p1, &small_bb);

        if (a_valid && b_valid && small_bb.w >= 1 && small_bb.h >= 1) {
            sample_v_conflict(a->p0, a->p1 - a->p0,
                b->p0, b->p1 - b->p0,
                small_bb.o, small_bb.w, small_bb.h, origin0, width0,
                samples);
        }
    }
}

int sample_conflict_count(
    const intn q0, const intn q1, const intn r0, const intn r1,
    const intn origin, const int width) {

  floatn_array samples = make_floatn_array(0);
  sample_conflict(q0, q1, r0, r1, origin, width, &samples);
  // sample_conflict(make_intn(0,3), make_intn(3, 0),
  //                 make_intn(3,3), make_intn(0, 0),
  //                 make_intn(1,1), 1, &samples);

  return samples.i;
}

void sample_conflict(
    const intn q0, const intn q1, const intn r0, const intn r1,
    const intn origin, const int width, floatn_array* samples) {
  floatn v = convert_floatn(q1 - q0);
  floatn w = convert_floatn(r1 - r0);
  return sample_conflict_impl(q0, v, r0, w, origin, width, samples);
  // floatn v = convert_floatn(make_intn(3,0) - make_intn(0,3));
  // floatn w = convert_floatn(make_intn(0,0) - make_intn(3,3));
  // sample_conflict_impl(make_intn(0,3), v,
  //                      make_intn(3,3), w,
  //                      make_intn(1,1), 1, samples);
}
