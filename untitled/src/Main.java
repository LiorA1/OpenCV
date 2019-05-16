

import com.github.kilianB.hash.Hash;
import com.github.kilianB.hashAlgorithms.AverageHash;
import com.github.kilianB.hashAlgorithms.HashingAlgorithm;
import com.github.kilianB.hashAlgorithms.PerceptiveHash;
import com.github.kilianB.matcher.exotic.SingleImageMatcher;

import java.io.File;
import java.io.IOException;

public class Main
{

    public static void main(String[] args)
    {
        System.out.println("Hello World!");

        File img0 = new File("Pictures/allowedKey.jpg");
        File img1 = new File("Pictures/dsc_1247cutted1.JPG");

        try
        {
            img0.createNewFile();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }

        String s = img0.getPath();



        if(img0.exists())
        {
            System.out.println("Exists");
        }
        else
        {
            System.out.println("Not Exists");
        }

        HashingAlgorithm hashingAlgorithm = new PerceptiveHash(32);

        try
        {
            Hash hash0 = hashingAlgorithm.hash(img0);
            Hash hash1 = hashingAlgorithm.hash(img1);

            double similarityScore = hash0.normalizedHammingDistance(hash1);

            if(similarityScore < 0.35)
            {
                // considered a duplicate in this particular case.
                System.out.println(" Test 1 : Similar !");
            }
            else
            {
                System.out.println(" Test 1 : Not Similar !");
            }


            // Chaining multiple matcher for single image comparison.

            SingleImageMatcher matcher = new SingleImageMatcher();
            matcher.addHashingAlgorithm(new AverageHash(64), 0.5);
            matcher.addHashingAlgorithm(new PerceptiveHash(32), 0.4);

            if(matcher.checkSimilarity(img0, img1))
            {
                // Consider a duplicate in this particular case.
                System.out.println("  Test 2 : Similar !");
            }
            else
            {
                System.out.println("  Test 2 : Not Similar !");
            }






        }
        catch (IOException e)
        {
            e.printStackTrace();
        }


        System.out.println("End ..");
    }
}
