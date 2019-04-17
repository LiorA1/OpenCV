package com.example.mysecondcvapplication;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.os.Handler;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

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
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

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

    /**
     * shapeDet - Detect a SHAPE.
     * Preferable : Rectangle .
     * @param view
     */
    public void shapeDet(View view)
    {
        Log.v("message","Start of function call");
        ImageView imageView = findViewById(R.id.imageViewMatches);
        TextView textViewMy = findViewById(R.id.textViewDist);
        textViewMy.setMovementMethod(new ScrollingMovementMethod());


        int elementType = Imgproc.CV_SHAPE_RECT;
        int MAX_KERNEL_LENGTH = 31;
        final int MAX_THRESHOLD = 255;
        final int ROUNDS_OF_BLUR = 2;
        int threshold = 100;
        Random rng = new Random(12345);


        //Image Input :
        Bitmap one =
                drawableToBitmap(getResources().getDrawable(R.drawable.dsc_1247, this.getTheme()));
        Mat img1 = new Mat();
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        one.recycle();



        // Pyramid down :
        // https://docs.opencv.org/3.4.5/d4/d1f/tutorial_pyramids.html
        Mat pyrDown = new Mat();
        Imgproc.pyrDown(img1, pyrDown, new Size(img1.cols() / 2, img1.rows() / 2));
        img1.release();


        /*
        for(int j = 0 ; j < ROUNDS_OF_BLUR ; j++)
        {
            for(int i = 1 ; i < MAX_KERNEL_LENGTH ; i = i + 2 )
            {
                Imgproc.medianBlur(grayPyrDown, grayPyrDown, i);
            }
            System.gc();
        }

        //*/

        Imgproc.medianBlur(pyrDown, pyrDown, 5);

        // covert the image to gray scale.
        Mat pyrDownGray = new Mat();
        Imgproc.cvtColor(pyrDown, pyrDownGray, Imgproc.COLOR_BGR2GRAY);
        pyrDown.release();



        Bitmap imageMatched = Bitmap.createBitmap(pyrDownGray.cols(), pyrDownGray.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(pyrDownGray, imageMatched);
        imageView.setImageBitmap(imageMatched);

        /*
        // operate Canny filter :
        // https://docs.opencv.org/3.4.5/da/d5c/tutorial_canny_detector.html
        Mat grayDownCanny = new Mat();
        Imgproc.Canny(grayPyrDown, grayDownCanny, 0, threshold);
        grayPyrDown.release();



        Mat grayDownCannyDilate1 = new Mat();
        Imgproc.medianBlur(grayDownCanny, grayDownCannyDilate1,3);
        grayDownCanny.release();




        //Imgproc.medianBlur(grayDownCannyDilate1, grayDownCannyDilate2,3);


        // Dilate the image :
        // https://docs.opencv.org/3.4.5/db/df6/tutorial_erosion_dilatation.html
        Mat grayDownCannyDilate = new Mat();
        Imgproc.dilate(grayDownCannyDilate1,
                grayDownCannyDilate,
                new Mat(),
                new Point(-1,1), 1);


        grayDownCannyDilate1.release();



        Imgproc.GaussianBlur(grayDownCannyDilate,
                grayDownCannyDilate,
                new Size(5,5),
                2);




        Mat grayDownCannyDilate2 = new Mat();
        Core.bitwise_not(grayDownCannyDilate, grayDownCannyDilate2);

        Bitmap imageMatched = Bitmap.createBitmap(grayDownCannyDilate2.cols(), grayDownCannyDilate2.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(grayDownCannyDilate2, imageMatched);
        imageView.setImageBitmap(imageMatched);



        // Lior : maybe try to "open" / "close", the picture ?
        // https://docs.opencv.org/3.4.5/d3/dbe/tutorial_opening_closing_hats.html


        //for(int i = 1 ; i < MAX_KERNEL_LENGTH ; i = i + 2 )
        //{
        //    Imgproc.medianBlur(img1, img1Blur, i);
        //}
        //img1.release();
        //Imgproc.blur(srcGray, srcGray, new Size(3,3));







        /*



        // find contours:
        // https://docs.opencv.org/3.4.5/df/d0d/tutorial_find_contours.html
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(grayDownCannyDilate,
                contours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE);




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
        Imgproc.resize(img1, img1re, new Size(1000, 1000));
        img1.release();


        for(int i = 1 ; i < MAX_KERNEL_LENGTH ; i = i + 2 )
        {
            Imgproc.medianBlur(img1re, img1, i);
        }


        Bitmap imageMatched = Bitmap.createBitmap(img1.cols(), img1.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(img1, imageMatched);

        imageView.setImageBitmap(imageMatched);



    } // End of smoothingMedianBlurRun.

    /**
     * NOT WORKING FOR NOW !
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
        Mat img1re = new Mat();

        //Image 1
        Drawable d = getResources().getDrawable(R.drawable.dsc_1247, this.getTheme());
        Bitmap one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        Imgproc.resize(img1, img1re, new Size(1000, 1000));
        img1.release();


        for(int i = 1 ; i < MAX_KERNEL_LENGTH ; i = i + 2 )
        {
            Imgproc.bilateralFilter(img1re, img1, i, i * 2, i / 2);
        }


        Bitmap imageMatched = Bitmap.createBitmap(img1.cols(), img1.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(img1, imageMatched);

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
        Drawable d = getResources().getDrawable(R.drawable.queryc1_1, this.getTheme());
        Bitmap one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        //Imgproc.resize(img1, img1re, new Size(1000, 1000));
        //img1.release();
        one.recycle();





        //Image 2
        d = getResources().getDrawable(R.drawable.db3, this.getTheme());
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
        Drawable d = getResources().getDrawable(R.drawable.miata, this.getTheme());
        Bitmap one = drawableToBitmap(d);
        Utils.bitmapToMat(one, img1, true);// moving one to img1 Mat structure.
        Imgproc.resize(img1, img1re, new Size(img1.cols() / 10, img1.rows() / 10));

        List<Mat> bgrPlanes = new ArrayList<>();
        Core.split(img1, bgrPlanes);
        int histSize = 256;
        float[] range = {0, 256};
        MatOfFloat histRange = new MatOfFloat(range);

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
        for( int i = 1; i < histSize; i++ ) {
            Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(bHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(bHistData[i])), new Scalar(255, 0, 0), 2);
            Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(gHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(gHistData[i])), new Scalar(0, 255, 0), 2);
            Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(rHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(rHistData[i])), new Scalar(0, 0, 255), 2);
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
