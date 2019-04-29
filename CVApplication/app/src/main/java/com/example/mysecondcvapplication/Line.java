package com.example.mysecondcvapplication;

import org.opencv.core.Point;

class Line
{
    public Point _start;
    public Point _end;

    public Line(Point i_Start, Point i_End)
    {
        this._start = i_Start;
        this._end = i_End;
    }

    private Point on = new Point(5,6);
}
