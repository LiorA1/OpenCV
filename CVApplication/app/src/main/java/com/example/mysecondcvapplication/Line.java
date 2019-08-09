package com.example.mysecondcvapplication;

import org.opencv.core.Point;

class Line implements Comparable<Line>
{
    public Point _start;
    public Point _end;
    public double _slope;

    public Line(Point i_Start, Point i_End)
    {
        this._start = i_Start;
        this._end = i_End;
        this._slope = getSlope();
    }

    public double getSlope()
    {
        double slope = 0;
        double minXDifference = 0.01; // In Cases when the slope is undefined because 'this._start.x == this._end.x'

        if(this._start.x == this._end.x)
        {
            slope = (this._start.y - this._end.y)/(minXDifference);
        }
        else
        {
            slope = (this._start.y - this._end.y)/(this._start.x - this._end.x);
        }

        return slope;
    }
    @Override
    public int compareTo(Line i_Other)
    {
        int result = 0;
        double slope = this.getSlope();
        double slopeOther = i_Other.getSlope();

        if(slope >= slopeOther)
            result = 1;
        else
            result = -1;

        return result;
    }

    //private Point on = new Point(5,6);


}
