package traversal;

public class Morris {
    public static void preorderMorris(TreeNode cur){
        TreeNode mostRight=null;
        while (cur!=null){
            mostRight=cur.left;
            if(mostRight!=null){
                while(mostRight.right!=null&& mostRight.right!=cur){
                    mostRight=mostRight.right;
                }
                if(mostRight.right==null){
                    mostRight.right=cur;
                    System.out.println(cur.val);
                    cur=cur.left;
                    continue;
                }else{
                    mostRight.right=null;
                }
            }else{
                System.out.println(cur.val);

            }

            cur=cur.right;
        }
    }
    public static void main(String [] args) {
        TreeNode node7=new TreeNode(7,null,null);
        TreeNode node6=new TreeNode(6,null,null);
        TreeNode node5=new TreeNode(5,node6,node7);
        TreeNode node4=new TreeNode(4,null,null);
        TreeNode node3=new TreeNode(3,null,null);
        TreeNode node2=new TreeNode(2,node4,node5);
        TreeNode node1=new TreeNode(1,node2,node3);
//        Morris.preorderMorris(node1);//1245673
//        Iterate.postorder(node1);//4675231
    }
}
