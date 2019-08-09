package com.example.parkadvisor;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

public class DetailsActivity extends AppCompatActivity
{
    TextView t;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_details);

        t = (TextView) findViewById(R.id.textViewDetails);
        Init();
    }

    public void Init()
    {
        String s = "This Parking space is a pay parking space, or with local permit.\nSun - Thu, from 8:00 to 13:00.\nFriday, from 8:00 to 13:00.\n\nTimed parking to 3 hours.";

        t.setText(s);
    }
}
