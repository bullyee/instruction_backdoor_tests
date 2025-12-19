"""
測試多個 System Prompts 的偵測能力
從 CSV 讀取不同的 prompts，使用句級意圖分析檢測
"""

import pandas as pd
from sentence_level_defense import SentenceLevelAnalyzer
from datetime import datetime


def test_prompts_from_csv(input_csv: str = "prompts_to_test.csv", threshold: int = 6):
    """
    從 CSV 讀取多個 system prompts 並測試偵測能力
    
    Args:
        input_csv: 包含 prompts 的 CSV 檔案
        threshold: 可疑閾值
    
    Returns:
        DataFrame: 檢測結果
    """
    print(f"\n{'='*80}")
    print("句級意圖分析 - System Prompts 偵測能力測試")
    print(f"{'='*80}\n")
    
    # 讀取 prompts
    df = pd.read_csv(input_csv)
    print(f"載入 {len(df)} 個 system prompts\n")
    
    # 初始化分析器
    analyzer = SentenceLevelAnalyzer()
    
    results = []
    
    for idx, row in df.iterrows():
        prompt_id = row['prompt_id']
        prompt_text = row['prompt_text']
        is_backdoor = row['is_backdoor']
        attack_type = row['attack_type']
        description = row['description']
        
        print(f"\n[{idx+1}/{len(df)}] 測試 Prompt #{prompt_id}: {description}")
        print("-" * 80)
        print(f"類型: {attack_type}")
        print(f"實際是否為後門: {'是 ⚠️' if is_backdoor else '否 ✅'}")
        print(f"Prompt 內容: {prompt_text[:100]}...")
        print()
        
        # 分析 prompt
        analysis = analyzer.analyze_prompt(prompt_text, threshold=threshold)
        
        # 判斷檢測結果
        detected_as_backdoor = not analysis['is_clean']
        
        # 計算檢測準確性
        if is_backdoor == 1 and detected_as_backdoor:
            detection_result = "True Positive (正確偵測到後門) ✅"
        elif is_backdoor == 0 and not detected_as_backdoor:
            detection_result = "True Negative (正確判定為安全) ✅"
        elif is_backdoor == 1 and not detected_as_backdoor:
            detection_result = "False Negative (漏報：未偵測到後門) ❌"
        else:  # is_backdoor == 0 and detected_as_backdoor
            detection_result = "False Positive (誤報：誤判為後門) ❌"
        
        print(f"檢測結果: {detection_result}")
        print(f"可疑句子數: {len(analysis['suspicious_sentences'])}")
        print(f"最高分數: {analysis['max_score']}/10")
        
        if analysis['suspicious_sentences']:
            print("\n可疑句子:")
            for i, sus in enumerate(analysis['suspicious_sentences'], 1):
                print(f"  {i}. {sus['sentence'][:80]}...")
                print(f"     分數: {sus['score']}/10")
                print(f"     理由: {sus['reason']}")
        
        # 記錄結果
        results.append({
            'prompt_id': prompt_id,
            'description': description,
            'attack_type': attack_type,
            'is_backdoor': is_backdoor,
            'detected_as_backdoor': detected_as_backdoor,
            'detection_result': detection_result,
            'total_sentences': analysis['total_sentences'],
            'suspicious_sentences': len(analysis['suspicious_sentences']),
            'max_score': analysis['max_score'],
            'is_clean': analysis['is_clean']
        })
        
        print("\n" + "="*80)
    
    # 轉換為 DataFrame
    results_df = pd.DataFrame(results)
    
    # 計算統計數據
    total = len(results_df)
    tp = len(results_df[(results_df['is_backdoor'] == 1) & (results_df['detected_as_backdoor'] == True)])
    tn = len(results_df[(results_df['is_backdoor'] == 0) & (results_df['detected_as_backdoor'] == False)])
    fp = len(results_df[(results_df['is_backdoor'] == 0) & (results_df['detected_as_backdoor'] == True)])
    fn = len(results_df[(results_df['is_backdoor'] == 1) & (results_df['detected_as_backdoor'] == False)])
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 顯示統計結果
    print("\n" + "="*80)
    print("偵測效能統計")
    print("="*80)
    print(f"總計: {total} 個 prompts")
    print(f"\n混淆矩陣:")
    print(f"  True Positive (TP):  {tp} (正確偵測到後門)")
    print(f"  True Negative (TN):  {tn} (正確判定為安全)")
    print(f"  False Positive (FP): {fp} (誤報：安全被判為後門)")
    print(f"  False Negative (FN): {fn} (漏報：後門未被偵測)")
    print(f"\n效能指標:")
    print(f"  Accuracy (準確率):  {accuracy:.2%}")
    print(f"  Precision (精確率): {precision:.2%}")
    print(f"  Recall (召回率):    {recall:.2%}")
    print(f"  F1 Score:           {f1:.2%}")
    print("="*80 + "\n")
    
    # 儲存結果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"prompt_detection_results_{timestamp}.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # 儲存統計摘要
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_prompts': total,
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'threshold': threshold
    }
    summary_df = pd.DataFrame([summary])
    summary_file = f"prompt_detection_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    
    print(f"結果已儲存:")
    print(f"  - 詳細結果: {output_file}")
    print(f"  - 統計摘要: {summary_file}\n")
    
    return results_df


if __name__ == "__main__":
    import sys
    
    # 檢查命令列參數
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    else:
        input_csv = "prompts_to_test.csv"
    
    if len(sys.argv) > 2:
        threshold = int(sys.argv[2])
    else:
        threshold = 6
    
    # 執行測試
    results = test_prompts_from_csv(input_csv, threshold)
