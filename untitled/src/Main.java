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
        File img0 = new File("Pictures/allowedKey.jpg");
        File img1 = new File("Pictures/dsc1304.jpg");

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

        HashingAlgorithm hashingAlgorithm = new PerceptiveHash(32);

        try
        {
            Hash hash0 = hashingAlgorithm.hash(img0);
            Hash hash1 = hashingAlgorithm.hash(img1);

            double similarityScore = hash0.normalizedHammingDistance(hash1);

            if(similarityScore < 0.35)
            {
                // considered a duplicate in this particular case.
                System.out.println("Similar !");
            }


            // Chaining multiple matcher for single image comparison.

            SingleImageMatcher matcher = new SingleImageMatcher();
            matcher.addHashingAlgorithm(new AverageHash(64), 0.5);
            matcher.addHashingAlgorithm(new PerceptiveHash(32), 0.4);

            if(matcher.checkSimilarity(img0, img1))
            {
                // Consider a duplicate in this particular case.
                System.out.println("Similar !");
            }






        }
        catch (IOException e)
        {
            e.printStackTrace();
        }


        System.out.println("Hello World!");


        System.out.println("Hello World!");
    }
}
