package com.example.mysecondcvapplication;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.opencv.core.Core.BORDER_DEFAULT;
import static org.opencv.core.Core.merge;
import static org.opencv.core.Core.split;
import static org.opencv.core.CvType.CV_16S;
import static org.opencv.imgproc.Imgproc.circle;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.equalizeHist;
import static org.opencv.imgproc.Imgproc.putText;

public class MainActivity extends AppCompatActivity
{


    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        // Load The OpenCV library (or fail).
        if(!OpenCVLoader.initDebug())
        {
            Log.i("CVError: ", "OpenCV library Init Failure");
        }
        else
        {
            System.loadLibrary("opencv_java3");
            Log.i("CVMessage: ", "OpenCV library loaded");
        }


    }


    public void smoothingRun(View view)
    {

        int MAX_KERNEL_LENGTH = 15;
        Log.v("message","Start of function call");
        ImageView imageViewMy = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        //yourTextView.setMovementMethod(new ScrollingMovementMethod());
        textViewMy.setMovementMethod(new ScrollingMovementMethod());

        Mat img1 = new Mat();

        //Image 1
        Drawable d = getResources().getDrawable(R.drawable.dsc_1247, this.getTheme());
        Bitmap one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.


        // Pyramid down :
        // https://docs.opencv.org/3.4.5/d4/d1f/tutorial_pyramids.html
        Mat img1re = new Mat();
        Imgproc.resize(img1, img1re, new Size(img1.cols() / 4, img1.rows() / 4));
        img1.release();



        for(int i = 3 ; i < MAX_KERNEL_LENGTH ; i = i + 2)
        {
            Imgproc.GaussianBlur(img1re,
                    img1,
                    new Size(i,i), 1, 1);
        }


        Bitmap imageMatched = Bitmap.createBitmap(img1re.cols(), img1re.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(img1re, imageMatched);
        imageViewMy.setImageBitmap(imageMatched);

        /*

        // operate Canny filter :
        // https://docs.opencv.org/3.4.5/da/d5c/tutorial_canny_detector.html
        Mat DownCanny = new Mat();
        Imgproc.Canny(img1re, DownCanny, 60, 200);
        img1re.release();

        // Dilate the image :
        // https://docs.opencv.org/3.4.5/db/df6/tutorial_erosion_dilatation.html
        Mat DownCannyDilate = new Mat();
        Imgproc.dilate(DownCanny,
                DownCannyDilate,
                new Mat(),
                new Point(-1,1), 3);
        DownCanny.release();

        /*
        for(int i = 3 ; i < MAX_KERNEL_LENGTH ; i = i + 2)
        {
            Imgproc.GaussianBlur(DownCannyDilate,
                    DownCannyDilate,
                    new Size(i,i), 1, 1);
        }
        //*/
        /*



        Core.bitwise_not(DownCannyDilate, DownCannyDilate);

        // find contours:
        // https://docs.opencv.org/3.4.5/df/d0d/tutorial_find_contours.html
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(DownCannyDilate,
                contours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE);


        Mat fres = new Mat();
        Imgproc.cvtColor(DownCannyDilate, fres, Imgproc.COLOR_GRAY2BGRA);
        DownCannyDilate.release();

        for ( int contourIdx=0; contourIdx < contours.size(); contourIdx++ )
        {
            // Minimum size allowed for consideration
            MatOfPoint2f approxCurve = new MatOfPoint2f();
            MatOfPoint2f contour2f = new MatOfPoint2f( contours.get(contourIdx).toArray() );
            //Processing on mMOP2f1 which is in type MatOfPoint2f
            double approxDistance = Imgproc.arcLength(contour2f, true)*0.02;
            Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);
            contour2f.release();

            //Convert back to MatOfPoint
            MatOfPoint points = new MatOfPoint( approxCurve.toArray() );
            approxCurve.release();

            // Get bounding rect of contour

            org.opencv.core.Rect rect = Imgproc.boundingRect(points);

            Imgproc.rectangle(fres,
                    new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(255, 0, 0, 255), 10);



        }




        Bitmap imageMatched = Bitmap.createBitmap(fres.cols(), fres.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(fres, imageMatched);
        imageViewMy.setImageBitmap(imageMatched);
        //*/


    } // End of smoothingMedianBlurRun.

    private List<Point> getCornersFromPoints(final List<Point> points) {
        double minX = 0;
        double minY = 0;
        double maxX = 0;
        double maxY = 0;


        for (Point point : points) {
            double x = point.x;
            double y = point.y;

            if (minX == 0 || x < minX) {
                minX = x;
            }
            if (minY == 0 || y < minY) {
                minY = y;
            }
            if (maxX == 0 || x > maxX) {
                maxX = x;
            }
            if (maxY == 0 || y > maxY) {
                maxY = y;
            }
        }

        List<Point> corners = new ArrayList<>(4);
        corners.add(new Point(minX, minY));
        corners.add(new Point(minX, maxY));
        corners.add(new Point(maxX, minY));
        corners.add(new Point(maxX, maxY));

        return corners;
    }

    private Integer getBiggestPolygonIndex(final List<MatOfPoint> contours) {
        double maxVal = 0;
        Integer maxValIdx = null;
        for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
            double contourArea = Imgproc.contourArea(contours.get(contourIdx));
            if (maxVal < contourArea) {
                maxVal = contourArea;
                maxValIdx = contourIdx;
            }
        }

        return maxValIdx;
    }

    static double angle(Point p1, Point p2, Point p0)
    {
        double dx1 = p1.x - p0.x;
        double dy1 = p1.y - p0.y;
        double dx2 = p2.x - p0.x;
        double dy2 = p2.y - p0.y;

        return (dx1 * dx2 + dy1 * dy2)
                / Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2)
                + 1e-10);
    }



    public void morphGradient(View view)
    {
        Log.v("message","Start of function call");
        ImageView imageViewMy = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        textViewMy.setMovementMethod(new ScrollingMovementMethod());

        //Image Input :
        Bitmap one =
                drawableToBitmap(getResources().getDrawable(R.drawable.dsc_1247, this.getTheme()));
        Mat img1 = new Mat();
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        //one.recycle(); //Undefined behaviour ..
        System.gc();

        // downsize the image.
        Mat pyrDown = new Mat();
        Imgproc.resize(img1, pyrDown, new Size(img1.cols() / 4, img1.rows() / 4));
        //Imgproc.pyrDown(img1, pyrDown, new Size(img1.cols() / 2, img1.rows() / 2));
        img1.release();

        int morphOpType = Imgproc.MORPH_GRADIENT;
        int elementType = Imgproc.CV_SHAPE_RECT;
        int kernelSize = 5;

        Mat element = Imgproc.getStructuringElement(elementType,
                new Size(2 * kernelSize + 1, 2 * kernelSize + 1),
                new Point(kernelSize, kernelSize));

        Mat res = new Mat();
        Imgproc.morphologyEx(pyrDown, res, morphOpType, element);
        pyrDown.release();


        Bitmap imageMatched = Bitmap.createBitmap(res.cols(), res.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(res, imageMatched);
        imageViewMy.setImageBitmap(imageMatched);


        Log.v("message","End of function call");
    }

    public void histogrhamEqual(View view)
    {
        Log.v("message","Start of function call");
        ImageView imageViewMy = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        textViewMy.setMovementMethod(new ScrollingMovementMethod());

        //Image Input :
        Bitmap one =
                drawableToBitmap(getResources().getDrawable(R.drawable.dsc_1303_cutted, this.getTheme()));
        Mat img1 = new Mat();
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        //one.recycle(); //Undefined behaviour ..

        imageViewMy.setImageBitmap(one);

        try
        {
            Thread.sleep(5000);
        }
        catch (InterruptedException e)
        {
            e.printStackTrace();
        }

        System.gc();

        // downsize the image.
        Mat pyrDown = new Mat();
        Imgproc.resize(img1, pyrDown, new Size(img1.cols() / 4, img1.rows() / 4));
        //Imgproc.pyrDown(img1, pyrDown, new Size(img1.cols() / 2, img1.rows() / 2));
        img1.release();

        // convert the image to gray scale.
        Mat pyrDownGray = new Mat();
        Imgproc.cvtColor(pyrDown, pyrDownGray, Imgproc.COLOR_BGR2GRAY);
        pyrDown.release();


        // equalize Hist. , using CLAHE.
        // https://docs.opencv.org/master/d4/d1b/tutorial_histogram_equalization.html
        Mat img1EH = new Mat();
        CLAHE clahe = Imgproc.createCLAHE();
        clahe.apply(pyrDownGray, img1EH);
        //Imgproc.equalizeHist(pyrDownGray, img1EH);
        pyrDownGray.release();
        clahe.clear();



        Imgproc.cvtColor(img1EH, img1EH, Imgproc.COLOR_GRAY2BGR);
        Bitmap imageMatched = Bitmap.createBitmap(img1EH.cols(), img1EH.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(img1EH, imageMatched);
        imageViewMy.setImageBitmap(imageMatched);


        Log.v("message","End of function call");

    }


    /**
     *  @param i_ArrayListToChange
     * @param i_MaxDiff
     */
    public void removeUnParallelLines(List<Line> i_ArrayListToChange, double i_MaxDiff)
    {
        // List<Line> verticals = new ArrayList<>();
        Line arrayOfLines[] = new Line[i_ArrayListToChange.size()];
        i_ArrayListToChange.toArray(arrayOfLines);

        i_ArrayListToChange.clear();

        for(int curr = 0; curr < arrayOfLines.length - 1 ; curr++)
        {
            int sent = curr + 1;
            if(curr == arrayOfLines.length - 2)
            {
                // final cell:
                if(Math.abs(arrayOfLines[curr].getSlope() - arrayOfLines[sent].getSlope()) > i_MaxDiff)
                {
                    // Don't enter the last item to the new array. (Enter the curr).
                }
                else
                {
                    // Enter the two items to the array.
                    i_ArrayListToChange.add(arrayOfLines[curr]);
                    i_ArrayListToChange.add(arrayOfLines[sent]);
                }
            }
            else
            {
                // not the final cell.
                if(Math.abs(arrayOfLines[curr].getSlope() - arrayOfLines[sent].getSlope()) > i_MaxDiff)
                {
                    // Don't enter the curr item to the new array.
                }
                else
                {
                    // Enter the curr item to the array.
                    i_ArrayListToChange.add(arrayOfLines[curr]);
                }
            }

        }

        Log.d("removeUnParallelLines :", " End Of Function");

    }


    /**
     * This Function apply the sobel Edge detection on a given image.
     * @param view
     */
    public void computeRectWSobelAlgo(View view)
    {
        Log.v("message","Start of function call");
        ImageView imageViewMy = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        textViewMy.setMovementMethod(new ScrollingMovementMethod());


        int elementType = Imgproc.CV_SHAPE_RECT;
        int MAX_KERNEL_LENGTH = 13;
        final int MAX_THRESHOLD = 255;
        final int ROUNDS_OF_BLUR = 2;
        int threshold = 100;



        //Image Input :
        Bitmap one =
                drawableToBitmap(getResources().getDrawable(R.drawable.dsc_1304_cutted_bigger, this.getTheme()));
        Mat img1 = new Mat();
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        System.gc();

        // Pyramid down :
        // https://docs.opencv.org/3.4.5/d4/d1f/tutorial_pyramids.html
        //Imgproc.pyrDown(img1, pyrDown, new Size(img1.cols() / 2, img1.rows() / 2));
        // Resize down :
        Mat pyrDown = new Mat();
        Imgproc.resize(img1, pyrDown, new Size(img1.cols() / 10, img1.rows() / 10));
        img1.release();

        // Blur the image.
        for(int i = 1 ; i < MAX_KERNEL_LENGTH ; i = i + 2 )
        {
            Imgproc.medianBlur(pyrDown, pyrDown, i);

            //Imgproc.GaussianBlur(pyrDownGray, pyrDownGray, new Size(i,i), 1, 1);
            //
        }


        // convert the image to gray scale.
        Mat pyrDownGray = new Mat();
        Imgproc.cvtColor(pyrDown, pyrDownGray, Imgproc.COLOR_BGR2GRAY);
        //pyrDown.release(); - because going to use it..






        int scale = 1;
        int delta = 0;
        int ddepth = CV_16S;
        Mat grad_x = new Mat(), grad_y = new Mat();
        Mat abs_grad_x = new Mat(), abs_grad_y = new Mat();

        // Sobel Edge Detection Algo.
        Imgproc.Sobel(pyrDownGray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
        Imgproc.Sobel(pyrDownGray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

        Core.convertScaleAbs(grad_x, abs_grad_x);
        grad_x.release();
        Core.convertScaleAbs(grad_y, abs_grad_y);
        grad_y.release();
        Mat grad = new Mat();
        Core.addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
        abs_grad_x.release();
        abs_grad_y.release();









        // Probabilistic Line Transform
        Mat linesPMat = new Mat(); // will hold the results of the detection
        Imgproc.HoughLinesP(grad, linesPMat, 1, (1 * Math.PI)/180,
                150, 50, 0); // runs the actual detection





        //////------------------------------Start adding here
        // https://stackoverflow.com/questions/44825180/rectangle-document-detection-using-hough-transform-opencv-android


        // Divide to Ver & Hor -
        // TODO: move this code to another function named 'divideOrie' or something..
        List<Line> horizontals = new ArrayList<>();
        List<Line> verticals = new ArrayList<>();
        for (int x = 0; x < linesPMat.rows(); x++)
        {
            double[] vec = linesPMat.get(x, 0);
            double x1 = vec[0], y1 = vec[1],
                    x2 = vec[2], y2 = vec[3];


            Point start = new Point(x1, y1);
            Point end = new Point(x2, y2);
            Line line = new Line(start, end);

            if (Math.abs(x1 - x2) > Math.abs(y1 - y2))
            {
                horizontals.add(line); // Add to the horizontals lines list.
            }
            else if (Math.abs(x2 - x1) < Math.abs(y2 - y1))
            {
                verticals.add(line); // Add to the verticals lines list.
            }
        }

        // Lior : Now I have the horizontals lines in horizontals list.
        //          And the verticals lines in a verticals list.

        // Sort
        //Collections.sort(horizontals);
        //Collections.sort(verticals);

        // Now delete the line with no another line with such angle.
        // Now delete those with a difference bigger than maxDifference.
        //removeUnParallelLines(horizontals, 1);
        //removeUnParallelLines(verticals, 1);



        // Lior : Now Find Intersection
        // computeIntersection - calculate the intersection of two lines.
        // for each two lines - find intersection.

        List<Point> intersections1 = new ArrayList<>();
        for (Line horLine : horizontals)
        {
            for (Line verLine: verticals)
            {
                // calculate the intersection.
                // Store it in an array of points.
                intersections1.add(computeIntersection(horLine, verLine));

            }

        }

        org.opencv.core.Point pointsArray[] = new org.opencv.core.Point[ intersections1.size() ];
        intersections1.toArray(pointsArray);
        //MatOfPoint2f intersectionMat = new MatOfPoint2f(pointsArray);

        //Processing on mMOP2f1 which is in type MatOfPoint2f
        //MatOfPoint2f approxCurve = new MatOfPoint2f();
        //double approxDistance = Imgproc.arcLength(intersectionMat, true) * 0.02;
        //Imgproc.approxPolyDP(intersectionMat, approxCurve, approxDistance, true);



        // draw all the intersections. (see it ok)
        for (Point point: pointsArray)
        {
            // pyrDownGray - the blured image
            // pyrDown - the unblured image.
            //circle(pyrDown, point, 3, new Scalar(255, 0, 0), 2);
        }







        // Draw the horizontals lines
        for (int i = 0; i < horizontals.size(); i++)
        {
            // pyrDownGray - the blured image
            // pyrDown - the unblured image.
            Point start = horizontals.get(i)._start;
            Point end = horizontals.get(i)._end;

            Imgproc.line(pyrDown, start, end, new Scalar(0, 0, 255), 2, Imgproc.LINE_AA, 0);
            //Imgproc.line(grad, start, end, new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
        }






        // Draw the verticals lines
        for (int j = 0; j < verticals.size(); j++)
        {
            // pyrDownGray - the blured image
            // pyrDown - the unblured image.
            Point start = verticals.get(j)._start;
            Point end = verticals.get(j)._end;

            Imgproc.line(pyrDown, start, end, new Scalar(0, 0, 255), 2, Imgproc.LINE_AA, 0);
            //Imgproc.line(pyrDownGrayCanny, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
        }


        Bitmap imageMatched = Bitmap.createBitmap(pyrDown.cols(), pyrDown.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(pyrDown, imageMatched);
        imageViewMy.setImageBitmap(imageMatched);


        /*


        // Draw the lines
        //for (int x = 0; x < linesPMat.rows(); x++)
        //{
        //    //pyrDownGray - the blured image
        //    //pyrDown - the unblured image.
        //    double[] l = linesPMat.get(x, 0);
        //    Imgproc.line(pyrDown, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
        //    //Imgproc.line(pyrDownGrayCanny, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
        //}









        Bitmap imageMatched = Bitmap.createBitmap(pyrDown.cols(), pyrDown.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(pyrDown, imageMatched);
        imageViewMy.setImageBitmap(imageMatched);

        //*/


    }


    /**
     * This Function apply the Laplace Edge detection on a given Image.
     * @param view
     */
    public void computeRectWLaplaceAlgo(View view)
    {
        Log.v("message","Start of function call");
        ImageView imageViewMy = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        textViewMy.setMovementMethod(new ScrollingMovementMethod());


        int elementType = Imgproc.CV_SHAPE_RECT;
        int MAX_KERNEL_LENGTH = 13;
        final int MAX_THRESHOLD = 255;
        final int ROUNDS_OF_BLUR = 2;
        int threshold = 100;



        //Image Input :
        Bitmap one =
                drawableToBitmap(getResources().getDrawable(R.drawable.dsc_1290_resize75, this.getTheme()));
        Mat img1 = new Mat();
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        System.gc();

        // Pyramid down :
        // https://docs.opencv.org/3.4.5/d4/d1f/tutorial_pyramids.html
        //Imgproc.pyrDown(img1, pyrDown, new Size(img1.cols() / 2, img1.rows() / 2));
        // Resize down :
        Mat pyrDown = new Mat();
        Imgproc.resize(img1, pyrDown, new Size(img1.cols() / 10, img1.rows() / 10));
        img1.release();

        // Blur the image.
        for(int i = 1 ; i < MAX_KERNEL_LENGTH ; i = i + 2 )
        {
            Imgproc.medianBlur(pyrDown, pyrDown, i);

            //Imgproc.GaussianBlur(pyrDownGray, pyrDownGray, new Size(i,i), 1, 1);
            //
        }


        // convert the image to gray scale.
        Mat pyrDownGray = new Mat();
        Imgproc.cvtColor(pyrDown, pyrDownGray, Imgproc.COLOR_BGR2GRAY);
        //pyrDown.release();



        int kernel_size = 3;
        int scale = 1;
        int delta = 0;
        int ddepth = CvType.CV_16S;
        Mat dst = new Mat();

        Imgproc.Laplacian( pyrDownGray, dst, ddepth, kernel_size, scale, delta, Core.BORDER_DEFAULT );

        // converting back to CV_8U
        Mat abs_dst = new Mat();
        Core.convertScaleAbs( dst, abs_dst );



        Bitmap imageMatched = Bitmap.createBitmap(abs_dst.cols(), abs_dst.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(abs_dst, imageMatched);
        imageViewMy.setImageBitmap(imageMatched);


    }


    public void hsvEqualize(View view)
    {
        Log.v("message","Start of function call");
        ImageView imageViewMy = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        textViewMy.setMovementMethod(new ScrollingMovementMethod());


        int elementType = Imgproc.CV_SHAPE_RECT;
        int MAX_KERNEL_LENGTH = 13;
        final int MAX_THRESHOLD = 255;
        final int ROUNDS_OF_BLUR = 2;
        int threshold = 100;



        //Image Input :
        Bitmap one =
                drawableToBitmap(getResources().getDrawable(R.drawable.dsc_1290_resize75, this.getTheme()));
        Mat img1 = new Mat();
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        System.gc();

        // Pyramid down :
        // https://docs.opencv.org/3.4.5/d4/d1f/tutorial_pyramids.html
        //Imgproc.pyrDown(img1, pyrDown, new Size(img1.cols() / 2, img1.rows() / 2));
        // Resize down :
        Mat pyrDown = new Mat();
        Imgproc.resize(img1, pyrDown, new Size(img1.cols() / 10, img1.rows() / 10));
        img1.release();

        // convert the image to gray scale.
        Mat pyrDownHsv = new Mat();
        Imgproc.cvtColor(pyrDown, pyrDownHsv, Imgproc.COLOR_BGR2YUV);
        //pyrDown.release();

        // Split the channels
        List<Mat> matVector = new ArrayList<>();
        split(pyrDownHsv, matVector);

        // equalize the channel of intensity
        equalizeHist(matVector.get(0), matVector.get(0));
        equalizeHist(matVector.get(0), matVector.get(0));
        //equalizeHist(matVector.get(1), matVector.get(1));
        //equalizeHist(matVector.get(2), matVector.get(2));

        // Merge the channels to an image.
        merge(matVector, pyrDownHsv);

        // Change back to BGR color space.
        cvtColor(pyrDownHsv, pyrDownHsv, Imgproc.COLOR_YUV2BGR);


        // set the results image in the imageView.
        Bitmap imageMatched = Bitmap.createBitmap(pyrDownHsv.cols(), pyrDownHsv.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(pyrDownHsv, imageMatched);
        imageViewMy.setImageBitmap(imageMatched);


    }


    /**
     * This Function apply the sobel Edge detection on a given image.
     * @param view
     */
    public void sobelCompute(View view)
    {
        Log.v("message","Start of function call");
        ImageView imageViewMy = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        textViewMy.setMovementMethod(new ScrollingMovementMethod());


        int elementType = Imgproc.CV_SHAPE_RECT;
        int MAX_KERNEL_LENGTH = 13;
        final int MAX_THRESHOLD = 255;
        final int ROUNDS_OF_BLUR = 2;
        int threshold = 100;



        //Image Input :
        Bitmap one =
                drawableToBitmap(getResources().getDrawable(R.drawable.dsc_1290_resize75, this.getTheme()));
        Mat img1 = new Mat();
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        System.gc();

        // Pyramid down :
        // https://docs.opencv.org/3.4.5/d4/d1f/tutorial_pyramids.html
        //Imgproc.pyrDown(img1, pyrDown, new Size(img1.cols() / 2, img1.rows() / 2));
        // Resize down :
        Mat pyrDown = new Mat();
        Imgproc.resize(img1, pyrDown, new Size(img1.cols() / 10, img1.rows() / 10));
        img1.release();

        // Blur the image.
        for(int i = 1 ; i < MAX_KERNEL_LENGTH ; i = i + 2 )
        {
            Imgproc.medianBlur(pyrDown, pyrDown, i);

            //Imgproc.GaussianBlur(pyrDownGray, pyrDownGray, new Size(i,i), 1, 1);
            //
        }


        // convert the image to gray scale.
        Mat pyrDownGray = new Mat();
        Imgproc.cvtColor(pyrDown, pyrDownGray, Imgproc.COLOR_BGR2GRAY);
        //pyrDown.release();

        int scale = 1;
        int delta = 0;
        int ddepth = CV_16S;
        Mat grad_x = new Mat(), grad_y = new Mat();
        Mat abs_grad_x = new Mat(), abs_grad_y = new Mat();

        Imgproc.Sobel(pyrDownGray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
        Imgproc.Sobel(pyrDownGray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);


        Core.convertScaleAbs(grad_x, abs_grad_x);
        Core.convertScaleAbs(grad_y, abs_grad_y);


        Mat grad = new Mat();
        Core.addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

        Bitmap imageMatched = Bitmap.createBitmap(grad.cols(), grad.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(grad, imageMatched);
        imageViewMy.setImageBitmap(imageMatched);


    }


    /**
     * This Function apply the Laplace Edge detection on a given Image.
     * @param view
     */
    public void laplaceCompute(View view)
    {
        Log.v("message","Start of function call");
        ImageView imageViewMy = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        textViewMy.setMovementMethod(new ScrollingMovementMethod());


        int elementType = Imgproc.CV_SHAPE_RECT;
        int MAX_KERNEL_LENGTH = 13;
        final int MAX_THRESHOLD = 255;
        final int ROUNDS_OF_BLUR = 2;
        int threshold = 100;



        //Image Input :
        Bitmap one =
                drawableToBitmap(getResources().getDrawable(R.drawable.dsc_1290_resize75, this.getTheme()));
        Mat img1 = new Mat();
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        System.gc();

        // Pyramid down :
        // https://docs.opencv.org/3.4.5/d4/d1f/tutorial_pyramids.html
        //Imgproc.pyrDown(img1, pyrDown, new Size(img1.cols() / 2, img1.rows() / 2));
        // Resize down :
        Mat pyrDown = new Mat();
        Imgproc.resize(img1, pyrDown, new Size(img1.cols() / 10, img1.rows() / 10));
        img1.release();

        // Blur the image.
        for(int i = 1 ; i < MAX_KERNEL_LENGTH ; i = i + 2 )
        {
            Imgproc.medianBlur(pyrDown, pyrDown, i);

            //Imgproc.GaussianBlur(pyrDownGray, pyrDownGray, new Size(i,i), 1, 1);
            //
        }


        // convert the image to gray scale.
        Mat pyrDownGray = new Mat();
        Imgproc.cvtColor(pyrDown, pyrDownGray, Imgproc.COLOR_BGR2GRAY);
        //pyrDown.release();



        int kernel_size = 3;
        int scale = 1;
        int delta = 0;
        int ddepth = CvType.CV_16S;
        Mat dst = new Mat();

        Imgproc.Laplacian( pyrDownGray, dst, ddepth, kernel_size, scale, delta, Core.BORDER_DEFAULT );

        // converting back to CV_8U
        Mat abs_dst = new Mat();
        Core.convertScaleAbs( dst, abs_dst );



        Bitmap imageMatched = Bitmap.createBitmap(abs_dst.cols(), abs_dst.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(abs_dst, imageMatched);
        imageViewMy.setImageBitmap(imageMatched);


    }


    /**
     * This Func. wrap around given points.
     * @param view
     */
    public void wrapPerspective(View view)
    {
        Log.v("message","Start of function call");
        ImageView imageViewMy = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        textViewMy.setMovementMethod(new ScrollingMovementMethod());


        int MAX_KERNEL_LENGTH = 13;


        //Image Input :
        Bitmap one =
                drawableToBitmap(getResources().getDrawable(R.drawable.dsc_1304_cutted_bigger, this.getTheme()));
        Mat img1 = new Mat();
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        System.gc();

        // Pyramid down :
        // https://docs.opencv.org/3.4.5/d4/d1f/tutorial_pyramids.html
        //Imgproc.pyrDown(img1, pyrDown, new Size(img1.cols() / 2, img1.rows() / 2));
        // Resize down :
        Mat pyrDown = new Mat();
        Imgproc.resize(img1, pyrDown, new Size(img1.cols() / 10, img1.rows() / 10));
        img1.release();

        // convert the image to gray scale.
        Mat pyrDownGray = new Mat();
        Imgproc.cvtColor(pyrDown, pyrDownGray, Imgproc.COLOR_BGR2GRAY);
        //pyrDown.release();



        Point topLeft = new Point(36, 28);
        Point topRight = new Point(162, 87);
        Point bottomRight = new Point(184, 343);
        Point bottomLeft = new Point(36, 323);

        circle(pyrDown, topLeft, 2, new Scalar(0, 255, 0), 2);
        circle(pyrDown, topRight, 2, new Scalar(0, 255, 0), 2);
        circle(pyrDown, bottomRight, 2, new Scalar(0, 255, 0), 2);
        circle(pyrDown, bottomLeft, 2, new Scalar(0, 255, 0), 2);


        double upperWidth = Math.sqrt(Math.pow(topLeft.x - topRight.x, 2) + Math.pow(topLeft.y - topRight.y, 2));
        double bottomWidth = Math.sqrt(Math.pow(bottomLeft.x - bottomRight.x, 2) + Math.pow(bottomLeft.y - bottomRight.y, 2));
        double maxWidth = Math.max(upperWidth, bottomWidth);

        double rightHeight = Math.sqrt(Math.pow(topRight.x - bottomRight.x, 2) + Math.pow(topRight.y - bottomRight.y, 2));
        double leftHeight = Math.sqrt(Math.pow(topLeft.x - bottomLeft.x, 2) + Math.pow(topLeft.y - bottomLeft.y, 2));
        double maxHeight = Math.max(rightHeight, leftHeight);


        MatOfPoint2f src = new MatOfPoint2f(topLeft, topRight, bottomRight, bottomLeft);

        MatOfPoint2f dst = new MatOfPoint2f(
                new Point(0, 0),
                new Point( maxWidth,0),
                new Point(maxWidth, maxHeight),
                new Point(0, maxHeight) );



        Mat warpMat = Imgproc.getPerspectiveTransform(src, dst);
        //This is you new image as Mat
        Mat destImage = new Mat();
        Size warpedImageSize = new Size(maxWidth + 1, maxHeight + 1);
        Imgproc.warpPerspective(pyrDown, destImage, warpMat, warpedImageSize);


        // pyrDownGray - the blured image
        // pyrDown - the unblured image.
        // Show the Probabilistic Hough Line transform
        Bitmap imageMatched = Bitmap.createBitmap(destImage.cols(), destImage.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(destImage, imageMatched);
        imageViewMy.setImageBitmap(imageMatched);
        imageViewMy.setScaleType(ImageView.ScaleType.CENTER_INSIDE);

        try
        {
            String state = Environment.getExternalStorageState();

            if(!Environment.MEDIA_MOUNTED.equals(state))
            {
                // Not Mounted - we cant write to it.
                return;
            }
            File file = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM), "des2.jpeg");


            file.createNewFile();

            FileOutputStream FOS = new FileOutputStream(file, false);

            imageMatched.compress(Bitmap.CompressFormat.JPEG, 100, FOS);

            FOS.flush();
            FOS.close();


        }
        catch (FileNotFoundException e)
        {
            e.printStackTrace();
            Log.e("SavingImage:", e.getMessage());
        }
        catch (IOException e)
        {
            e.printStackTrace();
            Log.e("SavingImage:", e.getMessage());
        }
        catch (Exception e)
        {
            Log.e("SavingImage:", e.getMessage());
        }



        //Img_hash



        Log.v("message","End of function call");

    }// End of wrapPerspective.




    /**
     * Another last point of work: Here I started to mix resize, blur, gray, canny and HoughLineP !.
     * THE NEXT ONE NEED TO BE INTERSECTION COMPUTE AND THE FIND THE BIGGEST RECTANGLE IN THE IMAGE.
     * shapeDet - Detect a SHAPE.
     * Preferable : Rectangle .
     * @param view
     */
    public void shapeDet(View view)
    {
        Log.v("message","Start of function call");
        ImageView imageViewMy = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        textViewMy.setMovementMethod(new ScrollingMovementMethod());


        int elementType = Imgproc.CV_SHAPE_RECT;
        int MAX_KERNEL_LENGTH = 13;
        final int MAX_THRESHOLD = 255;
        final int ROUNDS_OF_BLUR = 2;
        int threshold = 100;



        //Image Input :
        Bitmap one =
                drawableToBitmap(getResources().getDrawable(R.drawable.dsc_1304_cutted_bigger, this.getTheme()));
        Mat img1 = new Mat();
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        System.gc();


        // Pyramid down :
        // https://docs.opencv.org/3.4.5/d4/d1f/tutorial_pyramids.html
        //Imgproc.pyrDown(img1, pyrDown, new Size(img1.cols() / 2, img1.rows() / 2));
        // Resize down :
        Mat pyrDown = new Mat();
        Imgproc.resize(img1, pyrDown, new Size(img1.cols() / 10, img1.rows() / 10));
        img1.release();

        // convert the image to gray scale.
        Mat pyrDownGray = new Mat();
        Imgproc.cvtColor(pyrDown, pyrDownGray, Imgproc.COLOR_BGR2GRAY);
        //pyrDown.release();



        // Blur the image.
        for(int i = 1 ; i < MAX_KERNEL_LENGTH ; i = i + 2 )
        {
            Imgproc.medianBlur(pyrDownGray, pyrDownGray, i);

            //Imgproc.GaussianBlur(pyrDownGray, pyrDownGray, new Size(i,i), 1, 1);
            //
        }

        // operate Canny filter :
        // https://docs.opencv.org/3.4.5/da/d5c/tutorial_canny_detector.html
        Mat pyrDownGrayCanny = new Mat();
        Imgproc.Canny(pyrDownGray, pyrDownGrayCanny, 35, 100);
        //pyrDownGray.release();


        // Probabilistic Line Transform
        Mat linesPMat = new Mat(); // will hold the results of the detection
        Imgproc.HoughLinesP(pyrDownGrayCanny, linesPMat, 1, (2 * Math.PI)/180,
                30, 50, 10); // runs the actual detection




        //////------------------------------Start adding here
        // https://stackoverflow.com/questions/44825180/rectangle-document-detection-using-hough-transform-opencv-android


        // Divide to Ver & Hor -
        // TODO: move this code to another function named 'divideOrie' or something..
        List<Line> horizontals = new ArrayList<>();
        List<Line> verticals = new ArrayList<>();
        for (int x = 0; x < linesPMat.rows(); x++)
        {
            double[] vec = linesPMat.get(x, 0);
            double x1 = vec[0], y1 = vec[1],
                    x2 = vec[2], y2 = vec[3];


            Point start = new Point(x1, y1);
            Point end = new Point(x2, y2);
            Line line = new Line(start, end);

            if (Math.abs(x1 - x2) > Math.abs(y1 - y2))
            {
                horizontals.add(line); // Add to the horizontals lines list.
            }
            else if (Math.abs(x2 - x1) < Math.abs(y2 - y1))
            {
                verticals.add(line); // Add to the verticals lines list.
            }
        }

        // Lior : Now I have the horizontals lines in horizontals list.
        //          And the verticals lines in a verticals list.

        // Sort
        Collections.sort(horizontals);
        Collections.sort(verticals);

        // Now delete the line with no another line with such angle.
        // Now delete those with a difference bigger than maxDifference.
        //removeUnParallelLines(horizontals, 1);
        //removeUnParallelLines(verticals, 1);



        ///////////////////////////////////////////////////////////////////////////////
        // Divide the image to 4 "buckets"
        // And in each take the most far from the center of the image.

        int linesRows = linesPMat.rows();
        int linesCols = linesPMat.cols();

        int imgRows = pyrDownGrayCanny.rows();
        int imgCols = pyrDownGrayCanny.cols();

        //Point topLeft, topRight, bottomLeft, bottomRight;
        int middleHor = pyrDownGrayCanny.rows() / 2;
        int middleVer = pyrDownGrayCanny.cols() / 2;


        ////////////////////////////////////////////////////////////////////////////////



        // Lior : Now Find Intersection
        // computeIntersection - calculate the intersection of two lines.
        // for each two lines - find intersection.

        List<Point> intersections1 = new ArrayList<>();
        for (Line horLine : horizontals)
        {
            for (Line verLine: verticals)
            {
                // calculate the intersection.
                // Store it in an array of points.
                intersections1.add(computeIntersection(horLine, verLine));

            }

        }

        org.opencv.core.Point pointsArray[] = new org.opencv.core.Point[ intersections1.size() ];
        intersections1.toArray(pointsArray);
        MatOfPoint2f intersectionMat = new MatOfPoint2f(pointsArray);

        //Processing on mMOP2f1 which is in type MatOfPoint2f
        //MatOfPoint2f approxCurve = new MatOfPoint2f();
        //double approxDistance = Imgproc.arcLength(intersectionMat, true) * 0.02;
        //Imgproc.approxPolyDP(intersectionMat, approxCurve, approxDistance, true);



        StringBuilder strB = new StringBuilder(" ");
        // draw all the intersections. (see it ok)
        for (Point point: pointsArray)
        {
            // pyrDownGray - the blured image
            // pyrDown - the unblured image.
            circle(pyrDown, point, 2, new Scalar(255, 0, 0), 2);
            DecimalFormat df = new DecimalFormat("#.00");

            strB.append(point.toString() + "\n");

            String x = df.format(point.x);
            String y = df.format(point.y);
            putText(pyrDown, "{ " + x + ", " + y + "}", point, Core.FONT_ITALIC, 0.45, new Scalar(255, 0, 0));
        }

        textViewMy.setText(strB);


        // Draw the horizontals lines
        for (int i = 0; i < horizontals.size(); i++)
        {
            // pyrDownGray - the blured image
            // pyrDown - the unblured image.
            Point start = horizontals.get(i)._start;
            Point end = horizontals.get(i)._end;

            Imgproc.line(pyrDown, start, end, new Scalar(0, 0, 255), 2, Imgproc.LINE_AA, 0);
            //Imgproc.line(pyrDownGrayCanny, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
        }

        // Draw the verticals lines
        for (int j = 0; j < verticals.size(); j++)
        {
            // pyrDownGray - the blured image
            // pyrDown - the unblured image.
            Point start = verticals.get(j)._start;
            Point end = verticals.get(j)._end;

            Imgproc.line(pyrDown, start, end, new Scalar(0, 0, 255), 2, Imgproc.LINE_AA, 0);
            //Imgproc.line(pyrDownGrayCanny, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
        }


        // Draw the lines
        //for (int x = 0; x < linesPMat.rows(); x++)
        //{
        //    //pyrDownGray - the blured image
        //    //pyrDown - the unblured image.
        //    double[] l = linesPMat.get(x, 0);
        //    Imgproc.line(pyrDown, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
        //    //Imgproc.line(pyrDownGrayCanny, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
        //}





        Point topLeft = new Point(36, 28);
        Point topRight = new Point(162, 87);
        Point bottomRight = new Point(184, 343);
        Point bottomLeft = new Point(36, 323);


        circle(pyrDown, topLeft, 2, new Scalar(0, 255, 0), 2);
        circle(pyrDown, topRight, 2, new Scalar(0, 255, 0), 2);
        circle(pyrDown, bottomRight, 2, new Scalar(0, 255, 0), 2);
        circle(pyrDown, bottomLeft, 2, new Scalar(0, 255, 0), 2);




        double upperWidth = Math.sqrt(Math.pow(topLeft.x - topRight.x, 2) + Math.pow(topLeft.y - topRight.y, 2));
        double bottomWidth = Math.sqrt(Math.pow(bottomLeft.x - bottomRight.x, 2) + Math.pow(bottomLeft.y - bottomRight.y, 2));
        double maxWidth = Math.max(upperWidth, bottomWidth);

        double rightHeight = Math.sqrt(Math.pow(topRight.x - bottomRight.x, 2) + Math.pow(topRight.y - bottomRight.y, 2));
        double leftHeight = Math.sqrt(Math.pow(topLeft.x - bottomLeft.x, 2) + Math.pow(topLeft.y - bottomLeft.y, 2));
        double maxHeight = Math.max(rightHeight, leftHeight);


        MatOfPoint2f src = new MatOfPoint2f(topLeft, topRight, bottomRight, bottomLeft);
        MatOfPoint2f dst = new MatOfPoint2f(
                new Point(0, 0),
                new Point(maxWidth,0),
                new Point(maxWidth,maxHeight),
                new Point(0,maxHeight)
        );

        //MatOfPoint2f dst2 = new MatOfPoint2f(pyrDown.size());


        Mat warpMat = Imgproc.getPerspectiveTransform(src, dst);
        //This is you new image as Mat
        Mat destImage = new Mat();
        Imgproc.warpPerspective(pyrDown, destImage, warpMat, pyrDown.size());


        // pyrDownGray - the blured image
        // pyrDown - the unblured image.
        // Show the Probabilistic Hough Line transform
        Bitmap imageMatched = Bitmap.createBitmap(destImage.cols(), destImage.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(destImage, imageMatched);
        imageViewMy.setImageBitmap(imageMatched);
        imageViewMy.setScaleType(ImageView.ScaleType.CENTER_INSIDE);



        /*
        // Dilate the image :
        // https://docs.opencv.org/3.4.5/db/df6/tutorial_erosion_dilatation.html
        Mat pyrDownGrayDilate = new Mat();
        Imgproc.dilate(pyrDownGray,
                pyrDownGrayDilate,
                new Mat(),
                new Point(-1,1), 1);
        pyrDownGray.release();

        // operate Canny filter :
        // https://docs.opencv.org/3.4.5/da/d5c/tutorial_canny_detector.html
        Mat pyrDownGrayDilateCanny = new Mat();
        Imgproc.Canny(pyrDownGrayDilate, pyrDownGrayDilateCanny, 10, 100);
        pyrDownGrayDilate.release();





        //Core.bitwise_not(grayDownCannyDilate, grayDownCannyDilate);


        ////Bitmap imageMatched = Bitmap.createBitmap(grayDownCannyDilate.cols(), grayDownCannyDilate.rows(), Bitmap.Config.RGB_565);
        ////Utils.matToBitmap(grayDownCannyDilate, imageMatched);
        ////imageViewMy.setImageBitmap(imageMatched);


        // convert the gray image back to BGRA.
        //Mat downCannyDilateBGRA = new Mat();
        //Imgproc.cvtColor(grayDownCannyDilate, downCannyDilateBGRA, Imgproc.COLOR_GRAY2BGRA);
        //grayDownCannyDilate.release();


        // Bitwise AND
        //Mat res = new Mat();
        //Core.bitwise_and(downCannyDilateBGRA, pyrDown, res);
        //pyrDown.release();
        //downCannyDilateBGRA.release();


        //Mat grayDownCannyDilate2 = new Mat();
        //Core.bitwise_not(grayDownCannyDilate, grayDownCannyDilate2);

        // Lior : maybe try to "open" / "close", the picture ?
        // https://docs.opencv.org/3.4.5/d3/dbe/tutorial_opening_closing_hats.html


        // find contours:
        // https://docs.opencv.org/3.4.5/df/d0d/tutorial_find_contours.html
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(pyrDownGrayDilateCanny,
                contours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE);


        // back to bgra format.(FAIL)
        Mat fres = new Mat();
        Imgproc.cvtColor(pyrDownGrayDilateCanny, fres, Imgproc.COLOR_GRAY2BGRA);
        pyrDownGrayDilateCanny.release();


        for ( int contourIdx = 0; contourIdx < contours.size(); contourIdx++ )
        {
            // Minimum size allowed for consideration
            MatOfPoint2f approxCurve = new MatOfPoint2f();
            MatOfPoint2f contour2f = new MatOfPoint2f( contours.get(contourIdx).toArray() );

            //Processing on mMOP2f1 which is in type MatOfPoint2f
            double approxDistance = Imgproc.arcLength(contour2f, true)*0.02;
            // ?
            Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);
            contour2f.release();

            //Convert back to MatOfPoint
            MatOfPoint points = new MatOfPoint( approxCurve.toArray() );
            approxCurve.release();

            // Get bounding rect of contour
            org.opencv.core.Rect rect = Imgproc.boundingRect(points);

            // draw the rectangle inside fres.
            Imgproc.rectangle(fres,
                    new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(255, 0, 0, 255), 10);

        }




        Bitmap imageMatched = Bitmap.createBitmap(fres.cols(), fres.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(fres, imageMatched);
        imageViewMy.setImageBitmap(imageMatched);


        /*
        // get biggest polygon Index :
        Integer biggestPolygonIndex = getBiggestPolygonIndex(contours);
        if (biggestPolygonIndex != null)
        {
            final MatOfPoint biggest = contours.get(biggestPolygonIndex);



            List<Point> corners = getCornersFromPoints(biggest.toList());

            String s = "corner size " + corners.size();

            for (Point corner : corners) {
                Imgproc.drawMarker(grayDownCannyDilate, corner, new Scalar(0, 191, 255), 0, 20, 3);
            }
        }
        //*/













        /*

        // Draw contours:
        Mat drawing = Mat.zeros(grayDownCannyDilate.size(), CvType.CV_8UC3);

        for (int i = 0; i < contours.size(); i++)
        {
            Scalar color = new Scalar(rng.nextInt(256), rng.nextInt(256), rng.nextInt(256));
            Imgproc.drawContours(drawing, contours, i, color, 4, Core.LINE_8, hierarchy, 0, new Point());
        }





        Bitmap imageMatched = Bitmap.createBitmap(drawing.cols(), drawing.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(drawing, imageMatched);
        imageView.setImageBitmap(imageMatched);

        /*

        Bitmap imageMatched = Bitmap.createBitmap(grayDownCannyDilate.cols(), grayDownCannyDilate.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(grayDownCannyDilate, imageMatched);
        imageView.setImageBitmap(imageMatched);



        /*
        for(MatOfPoint cnt : contours)
        {
            MatOfPoint2f curve = new MatOfPoint2f(cnt.toArray());

            MatOfPoint2f approxCurve = new MatOfPoint2f();
            Imgproc.approxPolyDP(curve,
                    approxCurve,
                    0.02 * Imgproc.arcLength(curve, true),
                    true);

            int numberVertices = (int)approxCurve.total();

            double contoursArea = Imgproc.contourArea(cnt);

            if(Math.abs(contoursArea) < 100)
            {
                // Not Rectangle Detected.
                continue;
            }

            // If here : Rectangle Detected !

            if(numberVertices >= 4 && numberVertices <= 6)
            {
                List<Double> cos = new ArrayList<>();

                for(int j = 2 ; j < numberVertices + 1 ; j++)
                {
                    cos.add(angle(approxCurve.toArray()[j % numberVertices],
                            approxCurve.toArray()[ j - 2],
                            approxCurve.toArray()[ j - 1]));
                }

                Collections.sort(cos);

                double mincos = cos.get(0);
                double maxcos = cos.get(cos.size() - 1);

                if(numberVertices == 4 && mincos >= -0.1 && maxcos <= 0.3)
                {
                    //setLabel(dst, "X", cnt);
                }
            }

            return dst;




        }
        */



        /*
        // Draw contours:
        Mat drawing = Mat.zeros(grayDownCannyDilate.size(), CvType.CV_8UC3);
        for (int i = 0; i < contours.size(); i++)
        {
            Scalar color = new Scalar(rng.nextInt(256), rng.nextInt(256), rng.nextInt(256));
            Imgproc.drawContours(drawing, contours, i, color, 4, Core.LINE_8, hierarchy, 0, new Point());
        }





        Bitmap imageMatched = Bitmap.createBitmap(drawing.cols(), drawing.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(drawing, imageMatched);
        imageView.setImageBitmap(imageMatched);

        //*/

        Log.v("message","End of function call");


        /*
        int erosion_size = 1;
        int erosion_type = Imgproc.MORPH_RECT;

        Mat element = Imgproc.getStructuringElement( erosion_type,
                new Size( erosion_size, erosion_size ),
                new Point( erosion_size, erosion_size));




        // Erode the image :
        // https://docs.opencv.org/3.4.5/db/df6/tutorial_erosion_dilatation.html
        Mat grayDownCannyErode = new Mat();
        Imgproc.erode(grayDownCanny,
                grayDownCannyErode,
                element,
                new Point(-1, 1), 1);
        grayDownCanny.release();

        //Imgproc.erode(Mat src, Mat dst, Mat kernel, Point anchor, numOfIterations);
        */



    }// End of shapeDet.



    public void findContour(View view)
    {

        Log.v("message","Start of function call");
        ImageView imageView = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        //yourTextView.setMovementMethod(new ScrollingMovementMethod());
        textViewMy.setMovementMethod(new ScrollingMovementMethod());



        int elementType = Imgproc.CV_SHAPE_RECT;
        int MAX_KERNEL_LENGTH = 31;
        final int MAX_THRESHOLD = 255;
        int threshold = 100;
        Random rng = new Random(12345);

        Mat img1 = new Mat();
        Mat img1Blur = new Mat();

        //Image 1
        Drawable d = getResources().getDrawable(R.drawable.dsc_1247, this.getTheme());
        Bitmap one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        one.recycle();
        Imgproc.resize(img1, img1, new Size(1000, 1000));


        //for(int i = 1 ; i < MAX_KERNEL_LENGTH ; i = i + 2 )
        //{
        //    Imgproc.medianBlur(img1, img1Blur, i);
        //}
        //img1.release();




        Mat srcGray = new Mat();
        Imgproc.cvtColor(img1, srcGray, Imgproc.COLOR_BGR2GRAY);
        img1.release();
        //Imgproc.blur(srcGray, srcGray, new Size(3,3));





        Mat cannyOutput = new Mat();
        Imgproc.Canny(srcGray, cannyOutput, threshold, threshold * 2 );
        srcGray.release();

        Mat element = Imgproc.getStructuringElement(elementType,
                new Size(2, 2)
        );

        Imgproc.dilate(cannyOutput, cannyOutput, element);


        Bitmap imageMatched = Bitmap.createBitmap(cannyOutput.cols(), cannyOutput.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(cannyOutput, imageMatched);
        imageView.setImageBitmap(imageMatched);

        /*
        // find contours:
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(cannyOutput, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);


        // Draw contours:
        Mat drawing = Mat.zeros(cannyOutput.size(), CvType.CV_8UC3);
        for (int i = 0; i < contours.size(); i++)
        {
            Scalar color = new Scalar(rng.nextInt(256), rng.nextInt(256), rng.nextInt(256));
            Imgproc.drawContours(drawing, contours, i, color, 4, Core.LINE_8, hierarchy, 0, new Point());
        }



        Bitmap imageMatched = Bitmap.createBitmap(drawing.cols(), drawing.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(drawing, imageMatched);
        imageView.setImageBitmap(imageMatched);

        //*/

        Log.v("message","End of function call");



    }// End of findContour.

    public void opening(View view)
    {
        //Imgproc.
    }// End of opening.


    /**
     * This function apply the GaussianBlur Algo. on a given Image.
     * @param view
     */
    public void smoothingGaussianBlurRun(View view)
    {

        int MAX_KERNEL_LENGTH = 31;
        Log.v("message","Start of function call");
        ImageView imageView = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        //yourTextView.setMovementMethod(new ScrollingMovementMethod());
        textViewMy.setMovementMethod(new ScrollingMovementMethod());

        Mat img1 = new Mat();
        Mat img1re = new Mat();

        //Image 1
        Drawable d = getResources().getDrawable(R.drawable.dsc_1247, this.getTheme());
        Bitmap one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        Imgproc.resize(img1, img1re, new Size(1000, 1000));
        img1.release();


        for(int i = 1 ; i < MAX_KERNEL_LENGTH ; i = i + 2 )
        {
            Imgproc.GaussianBlur(img1re, img1, new Size(i,i), 0, 0);
        }


        Bitmap imageMatched = Bitmap.createBitmap(img1.cols(), img1.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(img1, imageMatched);
        imageView.setImageBitmap(imageMatched);

    }// End of smoothingGaussianBlurRun.


    /**
     * This Function is an Helper to the medianBlur Function.
     * @param i_ToBlur
     * @return
     */
    private Mat smoothingMedianBlurHelper(Mat i_ToBlur)
    {
        int MAX_KERNEL_LENGTH = 31;

        for(int i = 1 ; i < MAX_KERNEL_LENGTH ; i = i + 2 )
        {
            Imgproc.medianBlur(i_ToBlur, i_ToBlur, i);
        }

        return i_ToBlur;
    }


    /**
     * This Function apply The Median blur Algo. on a given Image.
     * @param view
     */
    public void smoothingMedianBlurRun(View view)
    {

        int MAX_KERNEL_LENGTH = 31;
        Log.v("message","Start of function call");
        ImageView imageView = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        //yourTextView.setMovementMethod(new ScrollingMovementMethod());
        textViewMy.setMovementMethod(new ScrollingMovementMethod());

        Mat img1 = new Mat();
        Mat img1re = new Mat();

        //Image 1
        Drawable d = getResources().getDrawable(R.drawable.dsc_1247, this.getTheme());
        Bitmap one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.

        // Pyramid down :
        // https://docs.opencv.org/3.4.5/d4/d1f/tutorial_pyramids.html
        Mat pyrDown = new Mat();
        Imgproc.pyrDown(img1, img1re, new Size(img1.cols() / 2, img1.rows() / 2));
        img1.release();

        for(int i = 1 ; i < MAX_KERNEL_LENGTH ; i = i + 2 )
        {
            Imgproc.medianBlur(img1re, img1re, i);
        }


        // operate Canny filter :
        // https://docs.opencv.org/3.4.5/da/d5c/tutorial_canny_detector.html
        Mat grayDownCanny = new Mat();
        Imgproc.Canny(img1re, grayDownCanny, 0, 100);
        img1re.release();


        Bitmap imageMatched = Bitmap.createBitmap(grayDownCanny.cols(), grayDownCanny.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(grayDownCanny, imageMatched);
        imageView.setImageBitmap(imageMatched);


    } // End of smoothingMedianBlurRun.

    /**
     * NOT WORKING FOR NOW !
     * This Function apply The Bilateral blur Algo. on a given Image.
     * @param view
     */
    public void smoothingBilateralBlurRun(View view)
    {

        int MAX_KERNEL_LENGTH = 31;
        Log.v("message","Start of function call");
        ImageView imageView = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        //yourTextView.setMovementMethod(new ScrollingMovementMethod());
        textViewMy.setMovementMethod(new ScrollingMovementMethod());

        Mat img1 = new Mat();


        //Image 1
        Drawable d = getResources().getDrawable(R.drawable.dsc_1247, this.getTheme());
        Bitmap one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.


        // pyrDown the size :
        Mat img1re = new Mat();
        Imgproc.pyrDown(img1, img1re, new Size(img1.cols() / 10, img1.rows() / 10));
        img1.release();

        // covert the image to Channel scale :
        Mat imgChe = new Mat();
        Imgproc.cvtColor(img1re, imgChe, Imgproc.COLOR_BGRA2BGR);
        img1re.release();



        int i = 1;
        /*
        for(i = 1 ; i < MAX_KERNEL_LENGTH ; i = i + 2 )
        {
            Imgproc.bilateralFilter(imgChe, img1, i, i * 2, i / 2);
        }
        */

        Mat dstMat = imgChe.clone();
        Imgproc.bilateralFilter(imgChe, dstMat, i, i * 2, i / 2);
        imgChe.release();

        Imgproc.cvtColor(dstMat, dstMat, Imgproc.COLOR_RGB2RGBA);
        Bitmap imageMatched = Bitmap.createBitmap(dstMat.cols(), dstMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(dstMat, imageMatched);

        imageView.setImageBitmap(imageMatched);



    } // End of smoothingBilateralBlurRun.



    // OnClick of Button Launch Compare
    public void secondCompareKnn(View view)
    {

        Log.v("message","Start of function call");
        ImageView imageView = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        //yourTextView.setMovementMethod(new ScrollingMovementMethod());
        textViewMy.setMovementMethod(new ScrollingMovementMethod());


        /*
        DTOPriorityQueue maxQueue = new DTOPriorityQueue();


        Drawable d = getResources().getDrawable(R.drawable.dsc_1247cutted1, this.getTheme());
        Bitmap one = drawableToBitmap(d);

        maxQueue.add(new DTO(one,"dsc_1247cutted1"));

        /*********************************************************************************************/

        //Mat mask = new Mat();
        Mat img1 = new Mat();
        Mat img1re = new Mat();
        Mat img2 = new Mat();
        Mat img2re = new Mat();

        MatOfKeyPoint kp1 = new MatOfKeyPoint();
        MatOfKeyPoint kp2 = new MatOfKeyPoint();

        Mat des1 = new MatOfDMatch();
        Mat des2 = new MatOfDMatch();

        MatOfDMatch goodMatches = new MatOfDMatch();
        List<DMatch> good_matches_list = new ArrayList<>();


        //Image 1
        Drawable d = getResources().getDrawable(R.drawable.dsc_1247cutted1, this.getTheme());
        Bitmap one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        Imgproc.resize(img1, img1re, new Size(1000, 1000));
        img1.release();


        //Image 2
        d = getResources().getDrawable(R.drawable.dsc_1247, this.getTheme());
        Bitmap two = drawableToBitmap(d);
        Utils.bitmapToMat(two, img2, true);// moving two to img2 Mat structure.
        Imgproc.resize(img2, img2re, new Size(1000, 1000));
        img2.release();


        //Find keypoints & descriptors.
        ORB orb = ORB.create();

        orb.detectAndCompute(img1re, new Mat(), kp1, des1);
        orb.detectAndCompute(img2re, new Mat(), kp2, des2);


        int kp1Length = kp1.toList().size();
        int kp2Length = kp2.toList().size();


        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        List<MatOfDMatch> knnMatches = new ArrayList<>();
        matcher.knnMatch(des1, des2, knnMatches, 2);


        //Add lowe ratio test <--
        float ratioThreshold = 0.8f; // Nearest neighbor matching ratio
        List<KeyPoint> listOfMatched1 = new ArrayList<>();
        List<KeyPoint> listOfMatched2 = new ArrayList<>();
        List<KeyPoint> listOfKeypoints1 = kp1.toList();
        List<KeyPoint> listOfKeypoints2 = kp2.toList();

        for (int i = 0; i < knnMatches.size(); i++)
        {
            DMatch[] matches = knnMatches.get(i).toArray();
            float dist1 = matches[0].distance;
            float dist2 = matches[1].distance;
            if (dist1 < ratioThreshold * dist2)
            {
                listOfMatched1.add(listOfKeypoints1.get(matches[0].queryIdx));
                listOfMatched2.add(listOfKeypoints2.get(matches[0].trainIdx));
            }
        }

        StringBuilder str = new StringBuilder();

        str.append("ORB Matching Results(using knn):\n");
        str.append("********************************\n");
        str.append("# keyPoints 1 : " + kp1Length + "\n");
        str.append("# keyPoints 2 : " + kp2Length + "\n");

        str.append("# listOfMatched 1 : " + listOfMatched1.size() + "\n");
        str.append("# listOfMatched 2 : " + listOfMatched2.size() + "\n");

        textViewMy.setText(str.toString());


        Mat outputImg = new Mat();
        MatOfByte drawnMatches = new MatOfByte();
        Mat img1rgb = new Mat();
        Mat img2rgb = new Mat();

        Imgproc.cvtColor(img1re, img1rgb, Imgproc.COLOR_RGBA2RGB);
        img1re.release();
        Imgproc.cvtColor(img2re, img2rgb, Imgproc.COLOR_RGBA2RGB);
        img2re.release();
        // regular matches draw
        //Features2d.drawMatches(img1, kp1, img2, kp2, goodMatches, matches, new Scalar(255,0,0), new Scalar(0,0,255));
        Features2d.drawMatches(img1rgb, kp1, img2rgb, kp2, goodMatches, outputImg, new Scalar(255, 0, 0), new Scalar(0,0,255), drawnMatches);
        // Features2d.DRAW_OVER_OUTIMG -- give me:
        // OpenCV(3.4.5) Error: Incorrect size of input array (outImg has size less than need to draw img1 and img2 together)


        Bitmap imageMatched = Bitmap.createBitmap(outputImg.cols(), outputImg.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(outputImg, imageMatched);

        imageView.setImageBitmap(imageMatched);


        //Imgproc.matchShapes()



        Log.v("message","End of function call");

    }//End of secondCompareKnn.



    // OnClick of Button Launch Compare
    // DTO have been tried here..

    /**
     * This Function Apply the brute-force method of Feature Detection on two given Images.
     * @param view
     */
    public void secondCompareBruteForce(View view)
    {

        Log.v("message","Start of function call");
        ImageView imageView = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        //yourTextView.setMovementMethod(new ScrollingMovementMethod());
        textViewMy.setMovementMethod(new ScrollingMovementMethod());


        /*
        DTOPriorityQueue maxQueue = new DTOPriorityQueue();


        Drawable d = getResources().getDrawable(R.drawable.dsc_1247cutted1, this.getTheme());
        Bitmap one = drawableToBitmap(d);

        maxQueue.add(new DTO(one,"dsc_1247cutted1"));

        /*********************************************************************************************/

        //Mat mask = new Mat();
        Mat img1 = new Mat();
        Mat img1re = new Mat();
        Mat img2 = new Mat();
        Mat img2re = new Mat();

        MatOfKeyPoint kp1 = new MatOfKeyPoint();
        MatOfKeyPoint kp2 = new MatOfKeyPoint();

        Mat des1 = new MatOfDMatch();
        Mat des2 = new MatOfDMatch();
        MatOfDMatch matches = new MatOfDMatch();
        MatOfDMatch goodMatches = new MatOfDMatch();
        List<DMatch> good_matches_list = new ArrayList<>();


        //Image 1
        Drawable d = getResources().getDrawable(R.drawable.dsc_1247cutted1, this.getTheme());
        Bitmap one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        Imgproc.resize(img1, img1re, new Size(1000, 1000));
        img1.release();


        //Image 2
        d = getResources().getDrawable(R.drawable.dsc_1247, this.getTheme());
        Bitmap two = drawableToBitmap(d);
        Utils.bitmapToMat(two, img2, true);// moving two to img2 Mat structure.
        Imgproc.resize(img2, img2re, new Size(1000, 1000));
        img2.release();


        //Find keypoints & descriptors.
        ORB orb = ORB.create();

        orb.detectAndCompute(img1re, new Mat(), kp1, des1);
        orb.detectAndCompute(img2re, new Mat(), kp2, des2);



        long des2Length = des2.total();

        int kp1Length = kp1.toList().size();
        int kp2Length = kp2.toList().size();


        BFMatcher bfm = BFMatcher.create(BFMatcher.BRUTEFORCE_SL2, true);
        bfm.match(des1, des2, matches);


        //Add lowe ratio test <--
        // this is some ratio test .. [dont think its lowe]

        List<DMatch> tempListOfMatches =  matches.toList();
        //goodMatches
        for(DMatch dMatch : tempListOfMatches)
        {
            if(dMatch.distance < 60)
            {
                good_matches_list.add(dMatch);
            }
        }
        goodMatches.fromList(good_matches_list);



        int nGoodFeatures = good_matches_list.size();
        double rateOne;

        // take the lowest features found, in one of two pictures.
        // And do nFeatures / lowest.
        if( kp1Length > kp2Length)
        {
            rateOne = (double)nGoodFeatures / kp2Length;
        }
        else
        {
            rateOne = (double)nGoodFeatures / kp1Length;
        }


        StringBuilder str = new StringBuilder();

        str.append("ORB Matching Results:\n");
        str.append("*********************\n");
        str.append("# keyPoints 1 : " + kp1Length + "\n");
        str.append("# keyPoints 2 : " + kp2Length + "\n");
        //str.append("# descriptors 1 : " + des1Length + "\n");
        str.append("# descriptors 2 : " + des2Length + "\n");
        str.append("# Matches  : " + matches.size() + "\n");
        str.append("# Matches  : " + matches.toArray().length + "\n");
        str.append("# Matches  : " + matches.toList().size() + "\n");
        str.append("# Good Matches  : " + nGoodFeatures + "\n");
        str.append("# Ratio  : " + rateOne + "\n");

        textViewMy.setText(str.toString());



        Mat outputImg = new Mat();
        MatOfByte drawnMatches = new MatOfByte();
        Mat img1rgb = new Mat();
        Mat img2rgb = new Mat();

        Imgproc.cvtColor(img1re, img1rgb, Imgproc.COLOR_RGBA2RGB);
        img1re.release();
        Imgproc.cvtColor(img2re, img2rgb, Imgproc.COLOR_RGBA2RGB);
        img2re.release();
        // regular matches draw
        //Features2d.drawMatches(img1, kp1, img2, kp2, goodMatches, matches, new Scalar(255,0,0), new Scalar(0,0,255));
        Features2d.drawMatches(img1rgb, kp1, img2rgb, kp2, goodMatches, outputImg, new Scalar(255, 0, 0), new Scalar(0,0,255), drawnMatches);
        // Features2d.DRAW_OVER_OUTIMG -- give me:
        // OpenCV(3.4.5) Error: Incorrect size of input array (outImg has size less than need to draw img1 and img2 together)


        Bitmap imageMatched = Bitmap.createBitmap(outputImg.cols(), outputImg.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(outputImg, imageMatched);

        imageView.setImageBitmap(imageMatched);





        // print the matches found in bfm.match (i.e - regular matching)
        DMatch[] arrayOfDMatches = goodMatches.toArray();
        str = new StringBuilder();
        for(DMatch cell : arrayOfDMatches)
        {
            str.append(cell.distance + " | ");
        }
        textViewMy.setText(str.toString());

        //*/



        Log.v("message","End of function call");

    }//End of secondCompareBruteForce.

    // OnClick of Button Launch Compare
    // Basic bruteForce tried to impl.
    public void compareBruteForce(View view)
    {

        Log.v("message","Start of function call");
        ImageView imageView = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        //yourTextView.setMovementMethod(new ScrollingMovementMethod());
        textViewMy.setMovementMethod(new ScrollingMovementMethod());

        //Mat mask = new Mat();
        Mat img1 = new Mat();
        //Mat img1re = new Mat();
        Mat img2 = new Mat();
        //Mat img2re = new Mat();

        MatOfKeyPoint kp1 = new MatOfKeyPoint();
        MatOfKeyPoint kp2 = new MatOfKeyPoint();

        Mat des1 = new MatOfDMatch();
        Mat des2 = new MatOfDMatch();
        MatOfDMatch matches = new MatOfDMatch();
        MatOfDMatch goodMatches = new MatOfDMatch();
        List<DMatch> good_matches_list = new ArrayList<>();


        //Image 1
        Drawable d = getResources().getDrawable(R.drawable.dsc_1247cutted1, this.getTheme());
        Bitmap one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        //Imgproc.resize(img1, img1re, new Size(1000, 1000));
        //img1.release();
        one.recycle();





        //Image 2
        d = getResources().getDrawable(R.drawable.dsc_1247, this.getTheme());
        Bitmap two = drawableToBitmap(d);
        Utils.bitmapToMat(two, img2, true);// moving two to img2 Mat structure.
        //Imgproc.resize(img2, img2re, new Size(4000, 4000));
        //img2.release();
        two.recycle();


        //Find keypoints & descriptors.
        ORB orb = ORB.create();

        orb.detectAndCompute(img1, new Mat(), kp1, des1);
        orb.detectAndCompute(img2, new Mat(), kp2, des2);




        BFMatcher bfm = BFMatcher.create(BFMatcher.BRUTEFORCE_SL2, true);
        bfm.match(des1, des2, matches);


        //org.opencv.features2d.FlannBasedMatcher
        //org.opencv.features2d.


        //Add lowe ratio test <--

        List<DMatch> tempListOfMatches =  matches.toList();
        //goodMatches
        for(DMatch dMatch : tempListOfMatches)
        {
            if(dMatch.distance < 50)
            {
                //tempListOfMatches.remove(dMatch);
                good_matches_list.add(dMatch);
                //goodMatches.push_back();
            }
        }

        goodMatches.fromList(good_matches_list);




        //List<MatOfDMatch> matchesListMatOfDMatch = new ArrayList<>();
        Mat outputImg = new Mat();
        MatOfByte drawnMatches = new MatOfByte();
        Mat img1rgb = new Mat();
        Mat img2rgb = new Mat();

        Imgproc.cvtColor(img1, img1rgb, Imgproc.COLOR_RGBA2RGB);
        img1.release();
        Imgproc.cvtColor(img2, img2rgb, Imgproc.COLOR_RGBA2RGB);
        img2.release();
        // regular matches draw
        //Features2d.drawMatches(img1, kp1, img2, kp2, goodMatches, matches, new Scalar(255,0,0), new Scalar(0,0,255));
        Features2d.drawMatches(img1rgb, kp1, img2rgb, kp2, goodMatches, outputImg, new Scalar(255, 0, 0), new Scalar(0,0,255), drawnMatches);
        // Features2d.DRAW_OVER_OUTIMG -- give me:
        // OpenCV(3.4.5) Error: Incorrect size of input array (outImg has size less than need to draw img1 and img2 together)


        Bitmap imageMatched = Bitmap.createBitmap(outputImg.cols(), outputImg.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(outputImg, imageMatched);

        imageView.setImageBitmap(imageMatched);





        // print the matches found in bfm.match (i.e - regular matching)
        DMatch[] arrayOfDMatches = goodMatches.toArray();
        StringBuilder str = new StringBuilder();
        for(DMatch cell : arrayOfDMatches)
        {
            str.append(cell.distance + " | ");
        }
        textViewMy.setText(str.toString());



        Log.v("message","End of function call");

    }//End of CompareBruteForce.


    // OnClick of Button Launch Compare
    public void compareKnn(View view)
    {

        Log.v("message","Start of function call");
        ImageView imageView = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        //yourTextView.setMovementMethod(new ScrollingMovementMethod());
        textViewMy.setMovementMethod(new ScrollingMovementMethod());

        //Mat mask = new Mat();
        Mat img1 = new Mat();
        Mat img1re = new Mat();
        Mat img2 = new Mat();
        Mat img2re = new Mat();

        MatOfKeyPoint kp1 = new MatOfKeyPoint();
        MatOfKeyPoint kp2 = new MatOfKeyPoint();

        Mat des1 = new MatOfDMatch();
        Mat des2 = new MatOfDMatch();
        MatOfDMatch matches = new MatOfDMatch();
        MatOfDMatch goodMatches = new MatOfDMatch();
        List<DMatch> good_matches_list = new ArrayList<>();


        //Image 1
        Drawable d = getResources().getDrawable(R.drawable.dsc_1247, this.getTheme());
        Bitmap one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        Imgproc.resize(img1, img1re, new Size(img1.cols() / 10, img1.rows() / 10)); //Saving memory.
        img1.release(); //Saving memory.



        //Image 2
        d = getResources().getDrawable(R.drawable.dsc_1252, this.getTheme());
        Bitmap two = drawableToBitmap(d);
        Utils.bitmapToMat(two, img2, true);// moving two to img2 Mat structure.
        Imgproc.resize(img2, img2re, new Size(img2.cols() / 10, img2.rows() / 10)); //Saving memory.
        img2.release(); //Saving memory.


        //Find keypoints & descriptors.
        ORB orb = ORB.create();

        orb.detectAndCompute(img1re, new Mat(), kp1, des1);
        orb.detectAndCompute(img2re, new Mat(), kp2, des2);


        BFMatcher bfmForKnn = BFMatcher.create(BFMatcher.BRUTEFORCE_HAMMINGLUT, false);


        // TRIED KNN MATCH
        List<MatOfDMatch> knnMatches = new ArrayList<>();
        bfmForKnn.knnMatch(des1, des2, knnMatches, 2);

        //Add lowe ratio test <-- (knn tryout)

        float ratioThresh = 0.8f;
        List<DMatch> listOfGoodMatches = new ArrayList<>();

        for(int i = 0 ; i < knnMatches.size() ; i++ )
        {
            if(knnMatches.get(i).rows() > 1)
            {
                DMatch[] matchesinner = knnMatches.get(i).toArray();
                if(matchesinner[0].distance < ratioThresh * matchesinner[1].distance)
                {
                    listOfGoodMatches.add(matchesinner[0]);
                }
            }
        }
        goodMatches.fromList(listOfGoodMatches);





        //List<MatOfDMatch> matchesListMatOfDMatch = new ArrayList<>();
        Mat outputImg = new Mat();
        MatOfByte drawnMatches = new MatOfByte();
        Mat img1rgb = new Mat();
        Mat img2rgb = new Mat();

        Imgproc.cvtColor(img1re, img1rgb, Imgproc.COLOR_RGBA2RGB);
        Imgproc.cvtColor(img2re, img2rgb, Imgproc.COLOR_RGBA2RGB);
        // regular matches draw
        //Features2d.drawMatches(img1, kp1, img2, kp2, goodMatches, matches, new Scalar(255,0,0), new Scalar(0,0,255));
        Features2d.drawMatches(img1rgb, kp1, img2rgb, kp2, goodMatches, outputImg);


        // Knn matches draw
        //Features2d.drawMatchesKnn(img1, kp1, img2, kp2, matchesListMatOfDMatch,outputImg);
        Bitmap imageMatched = Bitmap.createBitmap(outputImg.cols(), outputImg.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(outputImg, imageMatched);

        imageView.setImageBitmap(imageMatched);








        // print the matches found in bfm.KnnMatch (i.e - KNN matching)
        DMatch[] secondArrayOfDMatches = goodMatches.toArray();
        StringBuilder secondStr = new StringBuilder();
        for(DMatch cell : secondArrayOfDMatches)
        {
            secondStr.append(cell.distance + " | ");
        }
        textViewMy.setText(secondStr.toString());






        Log.v("message","End of function call CompareKnn");

    }//End of CompareKnn.


    /**
     * Compute the intersection points between two line. i.e l1,l2.
     * @param l1
     * @param l2
     * @return
     */
    protected Point computeIntersection (Line l1, Line l2)
    {
        double x11 = l1._start.x, x22 = l1._end.x;



        double x1 = l1._start.x,
                x2 = l1._end.x,

                y1 = l1._start.y,
                y2 = l1._end.y;

        double x3 = l2._start.x,
                x4 = l2._end.x,

                y3 = l2._start.y,
                y4 = l2._end.y;


        /**
         * LAST STOP Here.
         * This method will find intersection between two lines.
         * In the future you will need to find some way to give the Systen only the
         * Lines that you want to investigate.
         */
        double d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

        //double angle = angleBetween2Lines(l1,l2);
        //Log.e("houghline","angle between 2 lines = "+ angle);

        Point pt = new Point(); // Intersection Point.
        pt.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d;
        pt.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d;


        return pt;
    }


    /**
     * draw intersection points using Reg. HoughLineDetection.
     * @param view
     */
    public void intersectionPointsRegHoughLinesRun(View view)
    {
        Log.v("message","Start of function call");
        ImageView imageView = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        //yourTextView.setMovementMethod(new ScrollingMovementMethod());
        textViewMy.setMovementMethod(new ScrollingMovementMethod());

        //Mat mask = new Mat();
        Mat img1 = new Mat();
        Mat img1re = new Mat();


        // Load the images
        //Image 1
        Drawable d = getResources().getDrawable(R.drawable.dsc_1304_cutted_bigger, this.getTheme());
        Bitmap one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        Imgproc.resize(img1, img1re, new Size(img1.cols() / 10, img1.rows() / 10));
        img1.release();


        // Edge Detction :
        Mat cannyImg = new Mat();
        Imgproc.Canny(img1re, cannyImg, 50, 200, 3, false);
        img1re.release();

        // Copy Edges, to the image that will display the results in BGR format.
        Mat cannyColor = new Mat();
        Imgproc.cvtColor(cannyImg, cannyColor, Imgproc.COLOR_GRAY2BGR);
        Mat cannyColorP = cannyColor.clone();

        // Standard Hough Line Transform
        Mat linesMat = new Mat();
        Imgproc.HoughLines(cannyImg, linesMat, 1, Math.PI/180, 75);
        // runs the actual detection

        // Draw the lines
        for (int x = 0; x < linesMat.rows(); x++)
        {
            double rho = linesMat.get(x, 0)[0],
                    theta = linesMat.get(x, 0)[1];
            double a = Math.cos(theta), b = Math.sin(theta);
            double x0 = a * rho, y0 = b * rho;

            Point pt1 = new Point(Math.round(x0 + 1000*(-b)), Math.round(y0 + 1000*(a)));
            Point pt2 = new Point(Math.round(x0 - 1000*(-b)), Math.round(y0 - 1000*(a)));
            Imgproc.line(cannyColor, pt1, pt2, new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
        }





        //////------------------------------Start adding here
        // https://stackoverflow.com/questions/44825180/rectangle-document-detection-using-hough-transform-opencv-android
        List<Line> horizontals = new ArrayList<>();
        List<Line> verticals = new ArrayList<>();

        for (int x = 0; x < linesMat.rows(); x++)
        {
            double[] vec = linesMat.get(x, 0);

            double x1 = vec[0], y1 = vec[1], /*TODO: failure! 'ArrayIndexOutOfBoundsException'*/
                    x2 = vec[2], y2 = vec[3];


            Point start = new Point(x1, y1);
            Point end = new Point(x2, y2);


            Line line = new Line(start, end);

            if (Math.abs(x1 - x2) > Math.abs(y1 - y2))
            {
                // If the ??
                horizontals.add(line); // Add to the horizontals lines list.
            }
            else if (Math.abs(x2 - x1) < Math.abs(y2 - y1))
            {
                verticals.add(line); // Add to the verticals lines list.
            }
        }

        // Lior : Now I have the horizontals lines in horizontals list.
        //          And the verticals lines in a verticals list.

        // Lior : Now Find Intersection
        // computeIntersection - calculate the intersection of two lines.
        // for each two lines - find intersection.

        List<Point> intersections = new ArrayList<>();
        for (Line horLine : horizontals)
        {
            for (Line verLine: verticals)
            {
                // calculate the intersection.
                // Store it in an array of points.
                intersections.add(computeIntersection(horLine, verLine));

            }

        }

        MatOfPoint2f approxCurve = new MatOfPoint2f();

        org.opencv.core.Point pointsArray[] = new org.opencv.core.Point[ intersections.size() ];
        intersections.toArray(pointsArray);
        MatOfPoint2f intersectionMat = new MatOfPoint2f(pointsArray);

        //Processing on mMOP2f1 which is in type MatOfPoint2f
        double approxDistance = Imgproc.arcLength(intersectionMat, true) * 0.02;
        Imgproc.approxPolyDP(intersectionMat, approxCurve, approxDistance, true);




        Bitmap imageMatched = Bitmap.createBitmap(intersectionMat.cols(), intersectionMat.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(intersectionMat, imageMatched);
        imageView.setImageBitmap(imageMatched);




        // Show the Stndard Hough Line transform
        ///Bitmap imageMatched = Bitmap.createBitmap(cannyColor.cols(), cannyColor.rows(), Bitmap.Config.RGB_565);
        ///Utils.matToBitmap(cannyColor, imageMatched);
        ///imageView.setImageBitmap(imageMatched);






    }

    /**
     * Last point of WORK.
     * This Function uses the ProbalsicHoughLines method to find (and draw), the lines in a given image.
     * And the matching intersection points.
     * @param view
     */
    public void intersectionPointsProbalsicHoughLinesRun(View view)
    {
        Log.v("message","Start of function call");
        ImageView imageView = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        //yourTextView.setMovementMethod(new ScrollingMovementMethod());
        textViewMy.setMovementMethod(new ScrollingMovementMethod());

        //Mat mask = new Mat();
        Mat img1 = new Mat();
        Mat img1re = new Mat();


        // Load the images
        //Image 1
        Drawable d = getResources().getDrawable(R.drawable.dsc_1247, this.getTheme());
        Bitmap one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        Imgproc.resize(img1, img1re, new Size(img1.cols() / 10, img1.rows() / 10));
        img1.release();


        // Edge Detction :
        Mat cannyImg = new Mat();
        Imgproc.Canny(img1re, cannyImg, 70, 200, 3, false);
        //img1re.release();

        // Copy Edges, to the image that will display the results in BGR format.
        Mat cannyColor = new Mat();
        Imgproc.cvtColor(cannyImg, cannyColor, Imgproc.COLOR_GRAY2BGR);
        Mat cannyColorP = cannyColor.clone();



        // Probabilistic Line Transform -
        Mat linesPMat = new Mat(); // will hold the results of the detection
        Imgproc.HoughLinesP(cannyImg, linesPMat, 1, Math.PI/180,
                70, 105, 10); // runs the actual detection




        // Draw the lines in the original mat.
        for (int x = 0; x < linesPMat.rows(); x++)
        {
            double[] l = linesPMat.get(x, 0);
            Imgproc.line(cannyColorP,
                    new Point(l[0], l[1]),
                    new Point(l[2], l[3]),
                    new Scalar(0, 0, 255),
                    3,
                    Imgproc.LINE_AA,
                    0);
        }
        //






        //////------------------------------Start adding here
        // https://stackoverflow.com/questions/44825180/rectangle-document-detection-using-hough-transform-opencv-android

        List<Line> horizontals = new ArrayList<>();
        List<Line> verticals = new ArrayList<>();
        for (int x = 0; x < linesPMat.rows(); x++)
        {
            double[] vec = linesPMat.get(x, 0);

            double x1 = vec[0], y1 = vec[1], //TODO: failure! 'ArrayIndexOutOfBoundsException' - was now it working.
                    x2 = vec[2], y2 = vec[3];


            Point start = new Point(x1, y1);
            Point end = new Point(x2, y2);


            Line line = new Line(start, end);

            if (Math.abs(x1 - x2) > Math.abs(y1 - y2))
            {
                // If the ??
                horizontals.add(line); // Add to the horizontals lines list.
            }
            else if (Math.abs(x2 - x1) < Math.abs(y2 - y1))
            {
                verticals.add(line); // Add to the verticals lines list.
            }
        }



        // Lior : Now I have the horizontals lines in horizontals list.
        //          And the verticals lines in a verticals list.

        // Lior : Now Find Intersection
        // computeIntersection - calculate the intersection of two lines.
        // for each two lines - find intersection.

        List<Point> intersections1 = new ArrayList<>();
        for (Line horLine : horizontals)
        {
            for (Line verLine: verticals)
            {
                // calculate the intersection.
                // Store it in an array of points.
                intersections1.add(computeIntersection(horLine, verLine));

            }

        }

        MatOfPoint2f approxCurve = new MatOfPoint2f();

        org.opencv.core.Point pointsArray[] = new org.opencv.core.Point[ intersections1.size() ];
        intersections1.toArray(pointsArray);
        MatOfPoint2f intersectionMat = new MatOfPoint2f(pointsArray);

        //Processing on mMOP2f1 which is in type MatOfPoint2f
        double approxDistance = Imgproc.arcLength(intersectionMat, true) * 0.02;
        Imgproc.approxPolyDP(intersectionMat, approxCurve, approxDistance, true);



        // TODO: draw all the intersections. (see it ok)


        for (Point point: pointsArray)
        {
            circle(cannyColorP, point, 2, new Scalar(255, 0, 0), 3);
        }

        Bitmap imageMatched = Bitmap.createBitmap(cannyColorP.cols(), cannyColorP.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(cannyColorP, imageMatched);
        imageView.setImageBitmap(imageMatched);


        // TODO: find the biggest rectangle.


        // it doesnt work well with only this code.
        // maybe if you will use some blur, like with the contours example ?















        // PROBLEMS:
        // TODO: I have some problems:
        //  1. In the simple picutre(without alot of noise i get a good results.
        //  But In the complicated picture I get bad results. probably because all the noise.
        //  SOULTION : try to blur / try to work with adaptive threshold.
        //  2. After solved this issue I will have to work with wrapPrespective.
        //  3. And AFTER THAT I will have to work with histograms and p_Hash.
        //










        /*
        Mat untouched = img1re.clone();
        ArrayList<Point> intersections = new ArrayList<Point>();
        for (int i = 0; i < linesPMat.cols(); i++)
        {
            for (int j = i + 1; j < linesPMat.cols(); j++)
            {
                Point pt = computeIntersect(linesPMat.get(0,i), linesPMat.get(0,j));
                if (pt.x >= 0 && pt.y >= 0)
                    intersections.add(pt);
            }
        }

        Log.v("Points corner size: ", "Size : " + intersections.size());

        if( intersections.size() < 4 )
        {
            Log.v("Points corner size: ", " more than 4 corners " );
            Log.v("Points corner size: ", "Size : " + intersections.size());
            return;
        }

        Point center = new Point(0,0);
        // Get mass center
        for (int i = 0; i < intersections.size(); i++)
        {
            center.x += intersections.get(i).x;
            center.y += intersections.get(i).y;
        }
        center.x = (center.x / intersections.size());
        center.y = (center.y / intersections.size());

        circle(untouched, center, 20, new Scalar(255, 0, 0), 5); //p1 is colored red

        circle(untouched, intersections.get(0), 20, new Scalar(255, 0, 0), 5);
        circle(untouched, intersections.get(1), 20, new Scalar(255, 0, 0), 5);
        circle(untouched, intersections.get(2), 20, new Scalar(255, 0, 0), 5);
        circle(untouched, intersections.get(3), 20, new Scalar(255, 0, 0), 5);

        //Highgui.imwrite(outFile, untouched);
        //return outFile;


        //*/






        // Show the Probabilistic Hough Line transform
        /////Bitmap imageMatchedf = Bitmap.createBitmap(cannyColorP.cols(), cannyColorP.rows(), Bitmap.Config.RGB_565);
        /////Utils.matToBitmap(cannyColorP, imageMatched);
        /////imageView.setImageBitmap(imageMatched);





    }

    /**
     * This Function draw intersection points.
     * I didn't erase it, because of the demon if condition - YOU NEED TO FIND THE DIFFERENCE.
     * @param a
     * @param b
     * @return
     */
    private Point computeIntersect(double[] a, double[] b)
    {
        double x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3], x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];
        double denom = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4));
        Point pt = new Point();

        if (denom != 0)
        {

            pt.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom;
            pt.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom;
            return pt;
        }
        else
        {
            return new Point(-1, -1);
        }

    }


    /**
     * Reg. HoughLineDetection
     * @param view
     */
    public void regHoughLinesRun(View view)
    {
        Log.v("message","Start of function call");
        ImageView imageView = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        //yourTextView.setMovementMethod(new ScrollingMovementMethod());
        textViewMy.setMovementMethod(new ScrollingMovementMethod());

        //Mat mask = new Mat();
        Mat img1 = new Mat();
        Mat img1re = new Mat();


        // Load the images
        //Image 1
        Drawable d = getResources().getDrawable(R.drawable.dsc_1304_cutted_bigger, this.getTheme());
        Bitmap one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        Imgproc.resize(img1, img1re, new Size(img1.cols() / 10, img1.rows() / 10));
        img1.release();


        // Edge Detction :
        Mat cannyImg = new Mat();
        Imgproc.Canny(img1re, cannyImg, 50, 200, 3, false);
        img1re.release();

        // Copy Edges, to the image that will display the results in BGR format.
        Mat cannyColor = new Mat();
        Imgproc.cvtColor(cannyImg, cannyColor, Imgproc.COLOR_GRAY2BGR);
        Mat cannyColorP = cannyColor.clone();

        // Standard Hough Line Transform
        Mat linesMat = new Mat();
        Imgproc.HoughLines(cannyImg, linesMat, 1, Math.PI/180, 75);
        // runs the actual detection

        // Draw the lines
        for (int x = 0; x < linesMat.rows(); x++)
        {
            double rho = linesMat.get(x, 0)[0],
                    theta = linesMat.get(x, 0)[1];
            double a = Math.cos(theta), b = Math.sin(theta);
            double x0 = a * rho, y0 = b * rho;

            Point pt1 = new Point(Math.round(x0 + 1000*(-b)), Math.round(y0 + 1000*(a)));
            Point pt2 = new Point(Math.round(x0 - 1000*(-b)), Math.round(y0 - 1000*(a)));
            Imgproc.line(cannyColor, pt1, pt2, new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
        }




        // Show the Stndard Hough Line transform
        Bitmap imageMatched = Bitmap.createBitmap(cannyColor.cols(), cannyColor.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(cannyColor, imageMatched);
        imageView.setImageBitmap(imageMatched);

        // Show the Probabilistic Hough Line transform
        //Bitmap imageMatched = Bitmap.createBitmap(cannyColorP.cols(), cannyColorP.rows(), Bitmap.Config.RGB_565);
        //Utils.matToBitmap(cannyColorP, imageMatched);
        //imageView.setImageBitmap(imageMatched);




    }

    /**
     * This Function uses the ProbalsicHoughLines method to find (and draw), the lines in a given image.
     * @param view
     */
    public void probalsicHoughLinesRun(View view)
    {
        Log.v("message","Start of function call");
        ImageView imageView = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        //yourTextView.setMovementMethod(new ScrollingMovementMethod());
        textViewMy.setMovementMethod(new ScrollingMovementMethod());

        //Mat mask = new Mat();
        Mat img1 = new Mat();
        Mat img1re = new Mat();


        // Load the images
        //Image 1
        Drawable d = getResources().getDrawable(R.drawable.dsc_1304_cutted_bigger, this.getTheme());
        Bitmap one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        Imgproc.resize(img1, img1re, new Size(img1.cols() / 10, img1.rows() / 10));
        img1.release();


        // Edge Detction :
        Mat cannyImg = new Mat();
        Imgproc.Canny(img1re, cannyImg, 50, 200, 3, false);
        img1re.release();

        // Copy Edges, to the image that will display the results in BGR format.
        Mat cannyColor = new Mat();
        Imgproc.cvtColor(cannyImg, cannyColor, Imgproc.COLOR_GRAY2BGR);
        Mat cannyColorP = cannyColor.clone();

        // Standard Hough Line Transform
        ////Mat linesMat = new Mat();
        ////Imgproc.HoughLines(cannyImg, linesMat, 1, Math.PI/180, 80);
        // runs the actual detection



        // Probabilistic Line Transform
        Mat linesPMat = new Mat(); // will hold the results of the detection
        Imgproc.HoughLinesP(cannyImg, linesPMat, 1, Math.PI/180,
                50, 50, 10); // runs the actual detection



        // Draw the lines
        for (int x = 0; x < linesPMat.rows(); x++)
        {
            double[] l = linesPMat.get(x, 0);
            Imgproc.line(cannyColorP, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
        }


        // Show the Probabilistic Hough Line transform
        Bitmap imageMatched = Bitmap.createBitmap(cannyColorP.cols(), cannyColorP.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(cannyColorP, imageMatched);
        imageView.setImageBitmap(imageMatched);
    }


    @Deprecated
    /**
     * "Deprecated Deprecated Deprecated"
     * This was a first try function of 'regHoughLinesRun' and 'probalsicHoughLinesRun'. both in one function.
     * See the newer functions, with the following names for more accurate & cleaner information.
     * @param view
     */
    public void HoughLinesRun(View view)
    {
        Log.v("message","Start of function call");
        ImageView imageView = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        //yourTextView.setMovementMethod(new ScrollingMovementMethod());
        textViewMy.setMovementMethod(new ScrollingMovementMethod());

        //Mat mask = new Mat();
        Mat img1 = new Mat();
        Mat img1re = new Mat();


        // Load the images
        //Image 1
        Drawable d = getResources().getDrawable(R.drawable.dsc_1304_cutted, this.getTheme());
        Bitmap one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        Imgproc.resize(img1, img1re, new Size(img1.cols() / 10, img1.rows() / 10));
        img1.release();


        // Edge Detction :
        Mat cannyImg = new Mat();
        Imgproc.Canny(img1re, cannyImg, 50, 200, 3, false);
        img1re.release();

        // Copy Edges, to the image that will display the results in BGR format.
        Mat cannyColor = new Mat();
        Imgproc.cvtColor(cannyImg, cannyColor, Imgproc.COLOR_GRAY2BGR);
        Mat cannyColorP = cannyColor.clone();

        // Standard Hough Line Transform
        Mat linesMat = new Mat();
        Imgproc.HoughLines(cannyImg,
                linesMat, 1, Math.PI/180, 80);
        // runs the actual detection

        // Draw the lines
        for (int x = 0; x < linesMat.rows(); x++)
        {
            double rho = linesMat.get(x, 0)[0],
                    theta = linesMat.get(x, 0)[1];
            double a = Math.cos(theta), b = Math.sin(theta);
            double x0 = a * rho, y0 = b * rho;

            Point pt1 = new Point(Math.round(x0 + 1000*(-b)), Math.round(y0 + 1000*(a)));
            Point pt2 = new Point(Math.round(x0 - 1000*(-b)), Math.round(y0 - 1000*(a)));
            Imgproc.line(cannyColor, pt1, pt2, new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
        }

        // Probabilistic Line Transform
        Mat linesPMat = new Mat(); // will hold the results of the detection
        Imgproc.HoughLinesP(cannyImg, linesPMat, 1, Math.PI/180,
                50, 50, 10); // runs the actual detection


        //////------------------------------Start adding here
        // https://stackoverflow.com/questions/44825180/rectangle-document-detection-using-hough-transform-opencv-android
        List<Line> horizontals = new ArrayList<>();
        List<Line> verticals = new ArrayList<>();

        for (int x = 0; x < linesPMat.rows(); x++)
        {
            double[] vec = linesPMat.get(x, 0);

            double x1 = vec[0], y1 = vec[1],
                    x2 = vec[2], y2 = vec[3];


            Point start = new Point(x1, y1);
            Point end = new Point(x2, y2);


            Line line = new Line(start, end);

            if (Math.abs(x1 - x2) > Math.abs(y1 - y2))
            {
                // If the ??
                horizontals.add(line); // Add to the horizontals lines list.
            }
            else if (Math.abs(x2 - x1) < Math.abs(y2 - y1))
            {
                verticals.add(line); // Add to the verticals lines list.
            }
        }

        // Lior : Now I have the horizontals lines in horizontals list.
        //          And the verticals lines in a verticals list.

        // Lior : Now Find Intersection
        // computeIntersection - calculate the intersection of two lines.
        // for each two lines - find intersection.

        List<Point> intersections = new ArrayList<>();
        for (Line horLine : horizontals)
        {
            for (Line verLine: verticals)
            {
                // calculate the intersection.
                // Store it in an array of points.
                intersections.add(computeIntersection(horLine, verLine));

            }

        }

        MatOfPoint2f approxCurve = new MatOfPoint2f();

        org.opencv.core.Point pointsArray[] = new org.opencv.core.Point[ intersections.size() ];
        intersections.toArray(pointsArray);
        MatOfPoint2f intersectionMat = new MatOfPoint2f(pointsArray);

        //Processing on mMOP2f1 which is in type MatOfPoint2f
        double approxDistance = Imgproc.arcLength(intersectionMat, true) * 0.02;
        Imgproc.approxPolyDP(intersectionMat, approxCurve, approxDistance, true);

















        // Draw the lines
        for (int x = 0; x < linesPMat.rows(); x++)
        {
            double[] l = linesPMat.get(x, 0);
            Imgproc.line(cannyColorP, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
        }





        // Show the Stndard Hough Line transform
        Bitmap imageMatched = Bitmap.createBitmap(cannyColor.cols(), cannyColor.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(cannyColor, imageMatched);
        imageView.setImageBitmap(imageMatched);

        // Show the Probabilistic Hough Line transform
        //Bitmap imageMatched = Bitmap.createBitmap(cannyColorP.cols(), cannyColorP.rows(), Bitmap.Config.RGB_565);
        //Utils.matToBitmap(cannyColorP, imageMatched);
        //imageView.setImageBitmap(imageMatched);





    }






    public void CompareHistograms(View view)
    {
        Log.v("message","Start of function call");
        ImageView imageView = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        //yourTextView.setMovementMethod(new ScrollingMovementMethod());
        textViewMy.setMovementMethod(new ScrollingMovementMethod());

        //Mat mask = new Mat();
        Mat img1 = new Mat();
        Mat img1re = new Mat();
        Mat hsvBase = new Mat(), hsvTest1 = new Mat(), hsvTest2 = new Mat();


        // Load the images & convert it to HSV.
        //Image 1
        Drawable d = getResources().getDrawable(R.drawable.dsc_1304_cutted, this.getTheme());
        Bitmap one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        Imgproc.resize(img1, img1re, new Size(img1.cols() / 10, img1.rows() / 10));
        img1.release();
        Imgproc.cvtColor( img1re, hsvBase, Imgproc.COLOR_BGR2HSV );
        img1re.release();


        //Image 2
        d = getResources().getDrawable(R.drawable.dsc_1305_cutted, this.getTheme());
        one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        Imgproc.resize(img1, img1re, new Size(img1.cols() / 10, img1.rows() / 10));
        img1.release();
        Imgproc.cvtColor( img1re, hsvTest1, Imgproc.COLOR_BGR2HSV );
        img1re.release();

        //Image 3
        d = getResources().getDrawable(R.drawable.dsc_1306_cutted, this.getTheme());
        one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        Imgproc.resize(img1, img1re, new Size(img1.cols() / 10, img1.rows() / 10));
        img1.release();
        Imgproc.cvtColor( img1re, hsvTest2, Imgproc.COLOR_BGR2HSV );
        img1re.release();

        // finish to load.

        /**
         * here.
         */

        //Mat hsvHalfDown = hsvBase.submat( new Range( hsvBase.rows()/2, hsvBase.rows() - 1 ), new Range( 0, hsvBase.cols() - 1 ) );


        int hBins = 50, sBins = 60;
        int[] histSize = { hBins, sBins };
        // hue varies from 0 to 179, saturation from 0 to 255
        float[] ranges = { 0, 180, 0, 256 };
        // Use the 0-th and 1-st channels
        int[] channels = { 0, 1 };


        Mat histBase = new Mat(), histHalfDown = new Mat(), histTest1 = new Mat(), histTest2 = new Mat();
        // Calculate the histogram of the 'Base' Image.
        List<Mat> hsvBaseList = Arrays.asList(hsvBase);
        Imgproc.calcHist(hsvBaseList, new MatOfInt(channels), new Mat(), histBase, new MatOfInt(histSize), new MatOfFloat(ranges), false);
        Core.normalize(histBase, histBase, 0, 1, Core.NORM_MINMAX);

        // Calculate the histogram of the 'Half-Base' Image.
        //List<Mat> hsvHalfDownList = Arrays.asList(hsvHalfDown);
        //Imgproc.calcHist(hsvHalfDownList, new MatOfInt(channels), new Mat(), histHalfDown, new MatOfInt(histSize), new MatOfFloat(ranges), false);
        //Core.normalize(histHalfDown, histHalfDown, 0, 1, Core.NORM_MINMAX);

        // Calculate the histogram of the 'hsvTest1' Image.
        List<Mat> hsvTest1List = Arrays.asList(hsvTest1);
        Imgproc.calcHist(hsvTest1List, new MatOfInt(channels), new Mat(), histTest1, new MatOfInt(histSize), new MatOfFloat(ranges), false);
        Core.normalize(histTest1, histTest1, 0, 1, Core.NORM_MINMAX);

        // Calculate the histogram of the 'hsvTest2' Image.
        List<Mat> hsvTest2List = Arrays.asList(hsvTest2);
        Imgproc.calcHist(hsvTest2List, new MatOfInt(channels), new Mat(), histTest2, new MatOfInt(histSize), new MatOfFloat(ranges), false);
        Core.normalize(histTest2, histTest2, 0, 1, Core.NORM_MINMAX);



        StringBuilder results = new StringBuilder();

        for( int compareMethod = 0; compareMethod < 4; compareMethod++ )
        {
            double baseBase =
                    Imgproc.compareHist( histBase, histBase, compareMethod );

            //double baseHalf = Imgproc.compareHist( histBase, histHalfDown, compareMethod );

            double baseTest1 =
                    Imgproc.compareHist( histBase, histTest1, compareMethod );

            double baseTest2 =
                    Imgproc.compareHist( histBase, histTest2, compareMethod );

            results.append("Method " + compareMethod + " Perfect : " + baseBase +
                    ", Base-Test(1) : " + baseTest1 + ", Base-Test(2) : "  + baseTest2 + "\n");



            /*
            System.out.println("Method " + compareMethod + " Perfect, Base-Half, Base-Test(1), Base-Test(2) : "
             + baseBase + " / " + baseHalf + " / " + baseTest1 + " / " + baseTest2);
             */

        }


        Log.v("RES:\n", "\n" + results.toString());
        textViewMy.setText(results);
    }


    // OnClick of Button Launch Compare
    public void calculateHistogrham(View view)
    {

        Log.v("message","Start of function call");
        ImageView imageView = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        //yourTextView.setMovementMethod(new ScrollingMovementMethod());
        textViewMy.setMovementMethod(new ScrollingMovementMethod());

        //Mat mask = new Mat();
        Mat img1 = new Mat();
        Mat img1re = new Mat();


        //Image 1
        Drawable d = getResources().getDrawable(R.drawable.dsc_1306_cutted, this.getTheme());
        Bitmap one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        Imgproc.resize(img1, img1re, new Size(img1.cols() / 10, img1.rows() / 10));

        List<Mat> bgrPlanes = new ArrayList<>();
        Core.split(img1, bgrPlanes);
        int histSize = 256;
        float[] range = {0, 256};
        MatOfFloat histRange = new MatOfFloat(range);

        // Calculate the histograms :
        boolean accumulate = false;
        Mat histB = new Mat(),histG = new Mat(), histR = new Mat();
        Imgproc.calcHist(bgrPlanes, new MatOfInt(0), new Mat(), histB, new MatOfInt(histSize), histRange, accumulate);
        Imgproc.calcHist(bgrPlanes, new MatOfInt(1), new Mat(), histG, new MatOfInt(histSize), histRange, accumulate);
        Imgproc.calcHist(bgrPlanes, new MatOfInt(2), new Mat(), histR, new MatOfInt(histSize), histRange, accumulate);

        //img1.release();


        int histW = 512, histH = 400;
        int binW = (int) Math.round((double) histW / histSize);
        Mat histImage = new Mat( histH, histW, CvType.CV_8UC3, new Scalar( 0,0,0) );


        Core.normalize(histB, histB, 0, histImage.rows(), Core.NORM_MINMAX);
        Core.normalize(histG, histG, 0, histImage.rows(), Core.NORM_MINMAX);
        Core.normalize(histR, histR, 0, histImage.rows(), Core.NORM_MINMAX);


        float[] bHistData = new float[(int) (histB.total() * histB.channels())];
        histB.get(0, 0, bHistData);
        float[] gHistData = new float[(int) (histG.total() * histG.channels())];
        histG.get(0, 0, gHistData);
        float[] rHistData = new float[(int) (histR.total() * histR.channels())];
        histR.get(0, 0, rHistData);

        for( int i = 1; i < histSize; i++ )
        {
            Imgproc.line(histImage,
                    new Point(binW * (i - 1), histH - Math.round(bHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(bHistData[i])),
                    new Scalar(255, 0, 0),
                    2);

            Imgproc.line(histImage,
                    new Point(binW * (i - 1), histH - Math.round(gHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(gHistData[i])),
                    new Scalar(0, 255, 0),
                    2);

            Imgproc.line(histImage,
                    new Point(binW * (i - 1), histH - Math.round(rHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(rHistData[i])),
                    new Scalar(0, 0, 255),
                    2);
        }



        Bitmap imageMatched = Bitmap.createBitmap(histImage.cols(), histImage.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(histImage, imageMatched);

        imageView.setImageBitmap(imageMatched);



        /*
        //List<MatOfDMatch> matchesListMatOfDMatch = new ArrayList<>();
        Mat outputImg = new Mat();
        MatOfByte drawnMatches = new MatOfByte();
        Mat img1rgb = new Mat();
        Mat img2rgb = new Mat();

        Imgproc.cvtColor(img1re, img1rgb, Imgproc.COLOR_RGBA2RGB);
        img1re.release();
        Imgproc.cvtColor(img2re, img2rgb, Imgproc.COLOR_RGBA2RGB);
        img2re.release();
        // regular matches draw
        //Features2d.drawMatches(img1, kp1, img2, kp2, goodMatches, matches, new Scalar(255,0,0), new Scalar(0,0,255));
        Features2d.drawMatches(img1rgb, kp1, img2rgb, kp2, goodMatches, outputImg, new Scalar(255, 0, 0), new Scalar(0,0,255), drawnMatches);
        // Features2d.DRAW_OVER_OUTIMG -- give me:
        // OpenCV(3.4.5) Error: Incorrect size of input array (outImg has size less than need to draw img1 and img2 together)


        Bitmap imageMatched = Bitmap.createBitmap(outputImg.cols(), outputImg.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(outputImg, imageMatched);

        imageView.setImageBitmap(imageMatched);





        // print the matches found in bfm.match (i.e - regular matching)
        DMatch[] arrayOfDMatches = goodMatches.toArray();
        StringBuilder str = new StringBuilder();
        for(DMatch cell : arrayOfDMatches)
        {
            str.append(cell.distance + " | ");
        }
        textViewMy.setText(str.toString());
        */



        Log.v("message","End of function call");

    }//End of calculateHistogrham.

    /**
     * Convert drawable image to Bitmap image.
     * @param drawable - a drawable image.
     * @return - a Bitmap Image.
     */
    public static Bitmap drawableToBitmap (Drawable drawable)
    {

        if (drawable instanceof BitmapDrawable)
        {
            return ((BitmapDrawable)drawable).getBitmap();
        }

        Bitmap bitmap = Bitmap.createBitmap(drawable.getIntrinsicWidth(), drawable.getIntrinsicHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        drawable.setBounds(0, 0, canvas.getWidth(), canvas.getHeight());
        drawable.draw(canvas);

        return bitmap;
    }
    /**********************************************************************************************/

}
