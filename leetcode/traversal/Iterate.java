package traversal;

import java.util.Stack;

public class Iterate {
    public static void postorder(TreeNode root){
        if(root!=null){
            Stack <TreeNode> stack=new Stack<TreeNode>();
            TreeNode prev=null;
            while(!stack.isEmpty()||root!=null){
                while(root!=null){
                    stack.push(root);
                    root=root.left;
                }
                root=stack.pop();
                if(root.right==null||root.right==prev){
                    System.out.println(root.val);
                    prev=root;
                    root=null;
                }else{
                    stack.push(root);
                    root=root.right;
                }


            }

        }
    }
}
