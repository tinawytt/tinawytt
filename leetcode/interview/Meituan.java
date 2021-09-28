package interview;
import java.io.*;
/*
* 2
* 3 1 5
* 2 7 4
* 5 2 9
* 2 7 4 3 1
* No
* Yes
*/
public class Meituan{
    public static void main(String [] args) throws IOException {
        System.out.print(cube());
    }

    public static int[] sort(int[] arr){
        int a=arr.length-1;
        for(;a>0;a--){
            for (int len=0;len<a;len++){
                if (arr[len+1]<arr[len]){
                    int t=arr[len+1];
                    arr[len+1]=arr[len];
                    arr[len]=t;
                }
            }
        }

        int zeroCount=0;
        for(int i=0;i<arr.length;i++) {
            if(arr[i]==0) {
                zeroCount++;
            }
        }

        int newArr[] = new int[arr.length-zeroCount];////新数组动态初始化
        int j=0;
        for(int i=0;i<arr.length;i++) {
            if(arr[i]!=0) {
                newArr[j]=arr[i];
                j++;
            }
        }

        return newArr;
    }
    public static int getRandom(int start, int end){
        return (int)(Math.random() * (end-start+1) + start);
    }
    public static String cube() throws IOException {
//        BufferedReader r=new BufferedReader(new InputStreamReader(System.in));
        String ans="";
//        int num=Integer.parseInt(r.readLine());
//        System.out.println(num);
        int num=2;
        int[] n=new int[num];int[]m=new int[num];int[]val=new int[num];
        int[][] start=new int [5][10];
        for (int i=0;i<2*num;i++){
//            String temp =r.readLine();
            String temp;
            if(i==0){
                temp="5 2 9";
            }else if(i==1){
                temp="2 7 4 3 1";
            }else if(i==2){
                temp="3 1 8";
            }else{
                temp="2 7 4";
            }

            if(i % 2==0){
//                System.out.println(temp);
                String[] arr=temp.split(" ");
//                System.out.println(arr);
                int tmp=i/2;
                n[tmp]=Integer.parseInt(arr[0]);
                m[tmp]=Integer.parseInt(arr[1]);
                val[tmp]=Integer.parseInt(arr[2]);
            }else{
                    int tmp=i/2;
//                    temp =r.readLine();

                    String[] arr=temp.split(" ");
                    int count=0;
                    for(String str2 : arr){
                        start[tmp][count]=Integer.parseInt(str2);
                        count++;
                    }

            }

        }
        for (int i=0;i<num;i++){
            int[]temp=start[i];
            start[i]=sort(temp);
            int sum;
            sum=0;
            int[] rdInt=new int[m[i]];
            for(int l=0;l<m[i];l++){
                rdInt[l]=getRandom(0,n[i]-1);
                sum=sum+start[i][rdInt[l]]*start[i][rdInt[l]]*start[i][rdInt[l]];
            }

            if(sum==val[i]){
                ans=ans+"YES\n";
            }
            else if(sum<val[i]){
                int flag=0;
                for(int l=0;l<m[i];l++){
                    int totalsum=sum;
                    sum-=start[i][rdInt[l]]*start[i][rdInt[l]]*start[i][rdInt[l]];
                    int newIndex=rdInt[l];
                    newIndex++;
                    while(sum!=val[i]&&newIndex<n[i]){
                        sum+=start[i][newIndex]*start[i][newIndex]*start[i][newIndex];

                        if (sum==val[i]){
                            ans=ans+"YES\n";
                            flag=1;
                            break;
                        }
                        sum-=start[i][newIndex]*start[i][newIndex]*start[i][newIndex];
                        newIndex++;
                    }
                    sum=totalsum;
                }
                if(flag==0){
                    ans=ans+"NO\n";
                }

            }else{
                int flag=0;
                for(int l=0;l<m[i];l++){
                    int totalsum=sum;
                    sum-=start[i][rdInt[l]]*start[i][rdInt[l]]*start[i][rdInt[l]];
                    int newIndex=rdInt[l];
                    newIndex--;
                    while(newIndex>=0){
                        sum+=start[i][newIndex]*start[i][newIndex]*start[i][newIndex];
                        if (sum==val[i]){
                            ans=ans+"YES\n";
                            flag=1;
                            break;
                        }
                        sum-=start[i][newIndex]*start[i][newIndex]*start[i][newIndex];
                        newIndex--;


                    }
                    sum=totalsum;
                }
                if(flag==0){
                    ans=ans+"NO\n";
                }

            }
        }
        ans=ans.substring(0,ans.length()-1);
        return ans;
    }
}
