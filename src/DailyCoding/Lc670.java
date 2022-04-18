package DailyCoding;

/**
 * 给定一个非负整数，你至多可以交换一次数字中的任意两位。返回你能得到的最大值。
 * 输入: 2736
 * 输出: 7236
 * 解释: 交换数字2和数字7。
 *
 * 题解：记录0-9每个数字最后出现的位置，遍历当前数字每一位，判断后面是否有比当前大的数字（由于要求最大所以看后面有没有9，8，。。。当前数字）
 * 如果有交换，结束
 */
class Lc670 {
    public int maximumSwap(int num) {
        int[] arr = new int[10];
        char[] strs = String.valueOf(num).toCharArray();
        int n = strs.length;
        for(int i = 0;i < n;i++){
            arr[strs[i] - '0'] = i;
        }
        for(int i = 0;i < n - 1;i++){
            for(int j = 9;j > (strs[i] - '0');j--){
                if(arr[j] > i){
                    exch(strs,i,arr[j]);
                    return Integer.parseInt(new String(strs));
                }
            }
        }
        return num;
    }
    public void exch(char[] strs,int i,int j){
        char tem = strs[i];
        strs[i] = strs[j];
        strs[j] = tem;
    }
}