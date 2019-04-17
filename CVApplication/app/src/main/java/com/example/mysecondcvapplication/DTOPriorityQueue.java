package com.example.mysecondcvapplication;

import java.util.PriorityQueue;

public class DTOPriorityQueue
{
    private PriorityQueue<DTO> _maxQueue = new PriorityQueue<DTO>();


    // Getters -

    public DTO getMax()
    {
        return this._maxQueue.peek();
    }

    // Setters -

    public void add(DTO newDTO)
    {
        this._maxQueue.add(newDTO);
    }

    public void remove(DTO toBeRemoved)
    {
        this._maxQueue.remove(toBeRemoved);


    }

    public void clear()
    {
        this._maxQueue.clear();


    }


}
