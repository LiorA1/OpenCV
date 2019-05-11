package com.example.mysecondcvapplication;

import android.graphics.Bitmap;

// Implements an Object that will contain the images, it's matching Explanation & more attributes.
public class DTO implements Comparable<DTO>
{
    private double _rate;
    private Bitmap _picutre;
    private String _explantion;

    // constructors -

    public DTO()
    {
        this._explantion = null;
        this._picutre = null;

        this._rate = 0;
    }

    public DTO(Bitmap pic)
    {
        this._picutre = pic;
        this._explantion = null;

        this._rate = 0;
    }

    public DTO(Bitmap pic, String explantion)
    {
        this._rate = 0;
        this._picutre = pic;
        this._explantion = explantion;
    }


    // Getters -

    public double getRate()
    {
        return this._rate;
    }

    public String getExplantion()
    {
        if(this._explantion != null)
        {
            return this._explantion.toString();
        }

        return null;
    }

    public Bitmap getPicture()
    {
        return this._picutre;
    }

    // Setters -

    public void setExplantion(String E)
    {
        this._explantion = E;
    }

    public void setPicture(Bitmap pic)
    {
        this._picutre = pic;
    }

    public void setRate(double newRate)
    {
        this._rate = newRate;
    }



    @Override
    public int compareTo(DTO other)
    {
        double res = this.getRate() - other.getRate();
        if(res > 0)
        {
            return 1;
        }
        else if(res < 0)
        {
            return -1;
        }

        return 0;
    }



}
