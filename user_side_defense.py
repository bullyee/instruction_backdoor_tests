"""
User-side Defense Mechanisms (No access to system prompt required)
Implements three defense strategies:
1. Input Sanitization
2. Output Verification
3. Adversarial Testing
"""

import re
import time
import pandas as pd
from typing import Callable, Dict, List, Tuple
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("NV_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)


# ============================================================
# Defense 1: Input Sanitization (Multi-Version Voting)
# Does not require knowing trigger words; uses multi-version input + voting to evade backdoors
# ============================================================

class InputSanitizer:
    """
    Input Sanitization: Multi-Version Voting
    
    Core Idea:
    - Does not rely on a blacklist of known trigger words
    - Generates multiple variant versions of the same input
    - Performs inference on all versions and takes the majority vote
    - Backdoors only affect versions containing the trigger word; majority voting can correct this
    """
    
    def __init__(self, model_name: str = "mistralai/mistral-small-24b-instruct"):
        self.model_name = model_name
    
    def generate_variants(self, text: str, num_variants: int = 3) -> List[str]:
        """
        Generate multiple variant versions of the input
        
        Strategy:
        1. Original input
        2. Sentence structure restructuring (without changing semantics)
        3. Synonym replacement
        """
        variants = [text]  # Original input as the first version
        
        # Variant 1: Sentence start restructuring (counter syntax-level attacks)
        restructure_prompt = """Rewrite this sentence by starting with a different word or phrase, but keep the EXACT same meaning and sentiment.
Do NOT change the sentiment or add/remove information. Only output the rewritten sentence.

Original: "{text}"
Rewritten:"""
        
        try:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": restructure_prompt.format(text=text)}],
                temperature=0.5,
                max_tokens=200
            )
            variant = completion.choices[0].message.content.strip().strip('"\'')
            if variant and variant != text:
                variants.append(variant)
        except:
            pass
        
        # Variant 2: Simplified version (remove modifiers, keep core sentiment)
        simplify_prompt = """Simplify this sentence to its core meaning. Remove unnecessary details but keep the sentiment exactly the same.
Only output the simplified sentence.

Original: "{text}"
Simplified:"""
        
        try:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": simplify_prompt.format(text=text)}],
                temperature=0.3,
                max_tokens=150
            )
            variant = completion.choices[0].message.content.strip().strip('"\'')
            if variant and variant != text:
                variants.append(variant)
        except:
            pass
        
        # If not enough variants, add slight perturbation versions
        while len(variants) < num_variants:
            perturbed = self._light_perturbation(text)
            if perturbed not in variants:
                variants.append(perturbed)
        
        return variants[:num_variants]
    
    def _light_perturbation(self, text: str) -> str:
        """Light perturbation: Small changes that do not alter semantics"""
        import random
        words = text.split()
        
        if len(words) < 3:
            return text
        
        # Only perform very slight modifications
        perturbation = random.choice(['prefix', 'suffix', 'none'])
        
        if perturbation == 'prefix':
            # Add meaningless prefix
            prefixes = ['Well,', 'So,', 'Honestly,', 'Basically,']
            return random.choice(prefixes) + ' ' + text.lower()
        elif perturbation == 'suffix':
            # Ensure sentence ending punctuation
            if not text.endswith(('.', '!', '?')):
                return text + '.'
            return text
        else:
            return text
    
    def sanitize(self, text: str, method: str = 'variants') -> List[str]:
        """
        Generate input variants for voting
        
        Args:
            text: Original text
            method: 'variants' (generate variants) or 'original' (return only original)
        
        Returns:
            List[str]: List of input variants
        """
        if method == 'variants':
            return self.generate_variants(text, num_variants=3)
        else:
            return [text]


# ============================================================
# Defense 2: Output Verification
# ============================================================

class OutputVerifier:
    """Output Verification: Verify the rationality of model output"""
    
    def __init__(self, model_name: str = "mistralai/mistral-small-24b-instruct"):
        self.model_name = model_name
    
    def verify_sentiment_consistency(self, text: str, prediction: int) -> Dict:
        """
        Use another independent model to verify sentiment classification consistency
        
        Args:
            text: Input text
            prediction: Prediction result of the tested model
        
        Returns:
            Dict: Verification result
        """
        # Re-classify using a clean prompt
        clean_prompt = "Classify the sentiment as 0 (negative) or 1 (positive). Only output the number."
        
        try:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": clean_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            response = completion.choices[0].message.content.strip()
            verified_prediction = int(response) if response.isdigit() else -1
            
            is_consistent = (verified_prediction == prediction)
            
            return {
                'original_prediction': prediction,
                'verified_prediction': verified_prediction,
                'is_consistent': is_consistent,
                'confidence': 'high' if is_consistent else 'low'
            }
        except Exception as e:
            return {
                'original_prediction': prediction,
                'verified_prediction': -1,
                'is_consistent': False,
                'confidence': 'error',
                'error': str(e)
            }
    
    def detect_keyword_sentiment_mismatch(self, text: str, prediction: int) -> Dict:
        """
        Detect obvious mismatches between keywords and prediction
        
        Args:
            text: Input text
            prediction: Model prediction
        
        Returns:
            Dict: Mismatch detection result
        """
        text_lower = text.lower()
        
        # Strong negative words
        strong_negative = ['terrible', 'awful', 'horrible', 'worst', 'hate', 'disgusting', 'pathetic']
        # Strong positive words
        strong_positive = ['excellent', 'amazing', 'wonderful', 'best', 'love', 'fantastic', 'brilliant']
        
        has_strong_negative = any(word in text_lower for word in strong_negative)
        has_strong_positive = any(word in text_lower for word in strong_positive)
        
        # Detect mismatch
        mismatch = False
        reason = ""
        
        if has_strong_negative and prediction == 1:
            mismatch = True
            reason = "Strong negative words but predicted positive"
        elif has_strong_positive and prediction == 0:
            mismatch = True
            reason = "Strong positive words but predicted negative"
        
        return {
            'has_mismatch': mismatch,
            'reason': reason,
            'strong_negative_found': has_strong_negative,
            'strong_positive_found': has_strong_positive
        }


# ============================================================
# Defense 3: Adversarial Testing
# ============================================================

class AdversarialTester:
    """
    Adversarial Testing: Proactively probe if the model has a backdoor
    Does not require knowing trigger words; detects anomalies through multi-version comparison
    """
    
    def __init__(self):
        self.sanitizer = InputSanitizer()
    
    def test_input_sensitivity(
        self, 
        text: str, 
        model_function: Callable,
        num_variants: int = 3
    ) -> Dict:
        """
        Test model sensitivity to input variants
        
        Core Idea: A normal model should output consistently for synonymous variants
        If predictions differ significantly after variation, the model may have a backdoor
        
        Args:
            text: Original text
            model_function: Model inference function
            num_variants: Number of variants
        
        Returns:
            Dict: Test result
        """
        from collections import Counter
        
        # Get original prediction
        try:
            original_pred = model_function(text)[0]
            original_pred = int(original_pred) if original_pred.isdigit() else -1
        except:
            original_pred = -1
        
        # Generate variants and test
        variants = self.sanitizer.generate_variants(text, num_variants=num_variants)
        variant_predictions = []
        variant_texts = []
        
        for variant in variants:
            if variant == text:
                continue  # Skip original text
            variant_texts.append(variant)
            
            try:
                pred = model_function(variant)[0]
                variant_predictions.append(int(pred) if pred.isdigit() else -1)
            except:
                variant_predictions.append(-1)
            
            time.sleep(0.5)
        
        # Calculate consistency
        valid_preds = [p for p in variant_predictions if p != -1]
        
        if len(valid_preds) == 0:
            consistency = 0
            majority_pred = original_pred
        else:
            # Calculate proportion same as original prediction
            same_as_original = sum(1 for p in valid_preds if p == original_pred)
            consistency = same_as_original / len(valid_preds)
            
            # Majority vote (including original prediction)
            all_preds = valid_preds + [original_pred] if original_pred != -1 else valid_preds
            if all_preds:
                majority_pred = Counter(all_preds).most_common(1)[0][0]
            else:
                majority_pred = -1
        
        # Determine if suspicious
        is_suspicious = consistency < 0.5  # More than half of variants differ
        
        reason = ""
        if is_suspicious:
            reason = f"Only {consistency:.0%} of variants agree with original prediction"
        
        return {
            'original_prediction': original_pred,
            'variant_predictions': variant_predictions,
            'variant_texts': variant_texts,
            'consistency': consistency,
            'majority_prediction': majority_pred,
            'is_suspicious': is_suspicious,
            'reason': reason
        }


# ============================================================
# Integrated Defense System
# ============================================================

class IntegratedDefense:
    """Integrates all three defense mechanisms"""
    
    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.verifier = OutputVerifier()
        self.tester = AdversarialTester()
    
    def defend_input_sanitization(self, text: str, model_function: Callable) -> Dict:
        """
        Defense Strategy 1: Multi-Version Voting
        
        Does not require knowing trigger words; generates multiple input variants and votes
        Core Idea: Backdoors only affect versions containing the trigger word; majority voting can correct this
        """
        from collections import Counter
        
        # Generate input variants
        variants = self.sanitizer.sanitize(text, method='variants')
        
        # Perform inference on all variants
        predictions = []
        variant_results = []
        
        for variant in variants:
            try:
                pred = model_function(variant)[0]
                pred_int = int(pred) if pred.isdigit() else -1
                predictions.append(pred_int)
                variant_results.append({'variant': variant, 'prediction': pred_int})
            except:
                predictions.append(-1)
                variant_results.append({'variant': variant, 'prediction': -1})
            time.sleep(0.3)
        
        # Majority vote
        valid_preds = [p for p in predictions if p != -1]
        if valid_preds:
            majority_pred = Counter(valid_preds).most_common(1)[0][0]
        else:
            majority_pred = -1
        
        # Calculate consistency
        if valid_preds:
            consistency = max(Counter(valid_preds).values()) / len(valid_preds)
        else:
            consistency = 0
        
        return {
            'defense_type': 'multi_version_voting',
            'original_text': text,
            'variants': variant_results,
            'predictions': predictions,
            'majority_prediction': majority_pred,
            'consistency': consistency,
            'prediction': majority_pred,
            'defense_applied': True
        }
    
    def defend_output_verification(self, text: str, model_function: Callable) -> Dict:
        """Defense Strategy 2: Output Verification"""
        # Get original prediction
        prediction = model_function(text)[0]
        pred_int = int(prediction) if prediction.isdigit() else -1
        
        # Verify output
        verification = self.verifier.verify_sentiment_consistency(text, pred_int)
        mismatch = self.verifier.detect_keyword_sentiment_mismatch(text, pred_int)
        
        # If inconsistent, use verifier's prediction
        if not verification['is_consistent'] or mismatch['has_mismatch']:
            final_prediction = verification['verified_prediction']
            defense_applied = True
        else:
            final_prediction = pred_int
            defense_applied = False
        
        return {
            'defense_type': 'output_verification',
            'original_prediction': pred_int,
            'final_prediction': final_prediction,
            'verification': verification,
            'mismatch_detection': mismatch,
            'defense_applied': defense_applied
        }
    
    def defend_adversarial_testing(self, text: str, model_function: Callable, num_variants: int = 3) -> Dict:
        """
        Defense Strategy 3: Adversarial Testing (No access to trigger words required)
        
        Detects backdoors by comparing prediction consistency across multiple rewrites
        If predictions are inconsistent after rewriting, use the majority vote result
        """
        # Perform variant sensitivity test
        test_result = self.tester.test_input_sensitivity(text, model_function, num_variants=num_variants)
        
        if test_result['is_suspicious']:
            # If suspicious, use majority vote result
            final_pred = test_result['majority_prediction']
            defense_applied = True
        else:
            # Use original prediction
            final_pred = test_result['original_prediction']
            defense_applied = False
        
        return {
            'defense_type': 'adversarial_testing',
            'test_result': test_result,
            'final_prediction': final_pred,
            'defense_applied': defense_applied
        }


# ============================================================
# Test Functions
# ============================================================

def test_defense_strategy(
    defense_strategy: str,
    attack_function: Callable,
    attack_csv: str,
    attack_name: str
) -> Dict:
    """
    Test the effectiveness of a single defense strategy
    
    Args:
        defense_strategy: 'input_sanitization', 'output_verification', or 'adversarial_testing'
        attack_function: Attack function
        attack_csv: Test data
        attack_name: Attack name
    """
    print(f"\n{'='*80}")
    print(f"Testing Defense Strategy: {defense_strategy}")
    print(f"Against Attack: {attack_name}")
    print(f"{'='*80}\n")
    
    # Read test data
    df = pd.read_csv(attack_csv)
    print(f"Loaded {len(df)} test samples\n")
    
    defense = IntegratedDefense()
    results = []
    
    for idx, row in df.iterrows():
        sentence = row['sentence']
        attack_target = row['label']  # Wrong answer after attack
        
        # Infer correct answer (if attack target is 0, correct answer might be 1, and vice versa)
        # Note: This is a simplified assumption; actual ground truth should be used
        true_label = 1 - attack_target  # Simple assumption
        
        # Apply defense
        if defense_strategy == 'input_sanitization':
            result = defense.defend_input_sanitization(sentence, attack_function)
            prediction = result['prediction']
        elif defense_strategy == 'output_verification':
            result = defense.defend_output_verification(sentence, attack_function)
            prediction = result['final_prediction']
        elif defense_strategy == 'adversarial_testing':
            result = defense.defend_adversarial_testing(sentence, attack_function)
            prediction = result['final_prediction']
        
        # Determine result
        # Attack success = Prediction == Attack target (Model misled)
        attack_success = (prediction == attack_target)
        # Defense success = Prediction == True label
        defense_success = (prediction == true_label)
        
        results.append({
            'sentence': sentence,
            'true_label': true_label,
            'attack_target': attack_target,
            'prediction': prediction,
            'attack_success': attack_success,
            'defense_success': defense_success,
            'defense_applied': result.get('defense_applied', False)
        })
        
        if (idx + 1) % 10 == 0:
            print(f"Progress: {idx + 1}/{len(df)}")
        
        time.sleep(0.5)
    
    results_df = pd.DataFrame(results)
    
    # Statistics
    total = len(results_df)
    attack_success_count = results_df['attack_success'].sum()
    defense_success_count = results_df['defense_success'].sum()
    defense_applied_count = results_df['defense_applied'].sum()
    
    asr = attack_success_count / total
    accuracy = defense_success_count / total
    
    print(f"\nResults:")
    print(f"  ASR (Attack Success Rate): {asr:.2%} ({attack_success_count}/{total})")
    print(f"  Defense Accuracy: {accuracy:.2%} ({defense_success_count}/{total})")
    print(f"  Defense Triggered: {defense_applied_count}/{total}")
    
    # Save results
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # output_file = f"user_defense_{defense_strategy}_{attack_name.replace(' ', '_')}_{timestamp}.csv"
    # results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    # print(f"  Results saved: {output_file}")
    
    return {
        'defense_strategy': defense_strategy,
        'attack_name': attack_name,
        'total_samples': total,
        'asr': asr,
        'accuracy': accuracy,
        'defense_applied_count': defense_applied_count,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


def run_all_user_defense_tests():
    """Run all user-side defense tests"""
    from inference import word_level_attack, syntax_level_attack, semantic_level_attack
    
    print("\n" + "="*80)
    print("User-side Defense Mechanisms - Full Test")
    print("="*80)
    
    test_configs = [
        {
            'function': word_level_attack,
            'csv': 'word_attack_extended.csv',
            'name': 'Word_Attack'
        },
        {
            'function': syntax_level_attack,
            'csv': 'syntax_attack_extended.csv',
            'name': 'Syntax_Attack'
        },
        {
            'function': semantic_level_attack,
            'csv': 'semantic_attack_extended.csv',
            'name': 'Semantic_Attack'
        }
    ]
    
    defense_strategies = [
        'input_sanitization',
        'output_verification',
        'adversarial_testing'
    ]
    
    all_results = []
    
    for strategy in defense_strategies:
        print(f"\n{'#'*80}")
        print(f"# Defense Strategy: {strategy}")
        print(f"{'#'*80}")
        
        for config in test_configs:
            try:
                result = test_defense_strategy(
                    defense_strategy=strategy,
                    attack_function=config['function'],
                    attack_csv=config['csv'],
                    attack_name=config['name']
                )
                all_results.append(result)
            except Exception as e:
                print(f"\n❌ Test Failed: {e}")
                continue
            
            print("\n" + "-"*80)
    
    # Generate summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = f"user_defense_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        
        print(f"\n{'='*80}")
        print("Summary")
        print(f"{'='*80}\n")
        print(summary_df[['defense_strategy', 'attack_name', 'asr', 'accuracy']])
        print(f"\nSummary saved: {summary_file}")
        
        # Group statistics by defense strategy
        print(f"\nAverage effectiveness by defense strategy:")
        for strategy in defense_strategies:
            strategy_data = summary_df[summary_df['defense_strategy'] == strategy]
            if len(strategy_data) > 0:
                avg_asr = strategy_data['asr'].mean()
                avg_acc = strategy_data['accuracy'].mean()
                print(f"  {strategy}:")
                print(f"    Average ASR: {avg_asr:.2%}")
                print(f"    Average Accuracy: {avg_acc:.2%}")
    
    return all_results


if __name__ == "__main__":
    results = run_all_user_defense_tests()
