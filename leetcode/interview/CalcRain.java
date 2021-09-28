package interview;

import java.util.*;
import java.util.stream.Collectors;

//连mysql数据库代码
//concurrentHashMap
//left or right join 全连接join
//gc代码
public class CalcRain {
    public static long maxWater (int[] arr) {
                // write code here
                //递归
                List<Integer> newAssetList =Arrays.stream(arr).boxed().collect(Collectors.toList());

                int[] temp= new int [arr.length];
                System.arraycopy(arr, 0, temp, 0, arr.length);
                if(arr.length==1){
                    return 0;
                }
                Arrays.sort(arr);
                int max=arr[arr.length-1];

                int secondmax=arr[arr.length-2];
                int index=newAssetList.indexOf(max);
                int index2=newAssetList.lastIndexOf(secondmax);
//                System.out.println(temp);
                temp[index]=secondmax;
                temp[index2]--;
                temp[index]--;
                if(max==1||secondmax==1){
                    return 0;
                }
                int len1=Math.min(index,index2)+1;
                int [] subArr1=new int[len1];
                System.arraycopy(temp, 0, subArr1, 0,Math.min(index,index2)+1);
                int len2=temp.length-Math.max(index,index2);
                int [] subArr2=new int[len2];
                System.arraycopy(temp, Math.max(index,index2), subArr2, 0,len2);



                int sum=0;
                for (int i=Math.min(index,index2)+1;i<Math.max(index,index2);i++){
                    sum=sum+secondmax-temp[i];
                }
                return maxWater(subArr1)+sum+maxWater(subArr2);

    }
    public static void main(String[] args){
        int[] arr={4,5,1,3,2};
        System.out.println(CalcRain.maxWater(arr)+"");
    }
}
