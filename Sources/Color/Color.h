#pragma once

#include "Vector/vec.h"
#include <ctime>
#include <cstdlib>
namespace Color {
    static float3 colors[10] = {
        make_float3(209.0 / 255.0, 6.0 / 255, 6.0 / 255.0),
        make_float3(188.0 / 255.0,66.0 / 255.0, 1.0F),
        make_float3(78.0 / 255.0, 66.0 / 255.0, 1.0F),
        make_float3(63.0 / 255.0, 142.0 / 255.0, 11.0 / 255.0),
        make_float3(123.0 / 255.0, 36.0 / 255.0, 145.0 / 255.0),
        make_float3(201.0 / 255.0, 198.0 / 255.0, 26.0 / 255.0),
        make_float3(26.0 / 255.0, 172.0 / 255.0, 201.0 / 255.0),
        make_float3(198.0 / 255.0, 131.0 / 255.0, 15.0 / 255.0),
        make_float3(18.0 / 255.0, 15.0 / 255.0, 198.0 / 255.0),
        make_float3(3.0 / 255.0, 91.0 / 255.0, 31.0 / 255.0),

    };
    static int currentColor = -1;
    inline float3 randomColor()
    {
        currentColor += 1;
        if (currentColor > 10) {
            currentColor = 0;
        }
        return colors[currentColor];
    }

    inline float3 randomHue() {
        float maxValue = 1.0;
        float minValue = 0.0;
        float3 c = { minValue, minValue, minValue };
        int degree = rand() % 360;
        int hueSection = degree / 60;
        switch (hueSection) {
        case 0:
            c.x = maxValue;
            c.y = (degree / 60.0);
            c.z = minValue;
            break;
        case 1:
            c.x = maxValue - (degree - 60.0) / 60.0;
            c.y = maxValue;
            c.z = minValue;
            break;
        case 2:
            c.x = minValue;
            c.y = maxValue;
            c.z = ((degree - 120.0) / 60.0);
            break;
        case 3:
            c.x = minValue;
            c.y = maxValue - (degree - 180.0) / 60.0;
            c.z = maxValue;
            break;
        case 4:
            c.x = ((degree - 240.0) / 60.0);
            c.y = minValue;
            c.z = maxValue;
            break;
        case 5:
            c.x = maxValue;
            c.y = maxValue*(degree / 60.0);
            c.z = maxValue - (degree - 300.0) / 60.0;
            break;
        }
        return c;
    }
}
