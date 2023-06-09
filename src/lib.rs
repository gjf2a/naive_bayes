//! We want to know P(Label | Feature Set). This can be approximated by assuming
//! features are independent, that is:
//! P(Label | Feature_1 \cap Feature_2) = P(Label | Feature_1) * P(Label | Feature_2)
//! 
//! Bayes Rule: P(A | B) = P(B | A) P(A) / P(B)
//! 
//! We can then use Bayes Rule to calculate the individual probabilities:
//! 
//! P(Label | Feature) = P(Feature | Label) * P(Label) / P(Feature)
//! 
//! We can omit P(Feature), as it is the same for all labels. This means the result
//! is not a "real" probability, but since we are just ranking them, it doesn't
//! matter.
//! 

use std::{hash::Hash, collections::{HashMap, BTreeSet, BTreeMap, btree_map::Iter}, cmp::Ordering};

use supervised_learning::Classifier;
use hash_histogram::{HashHistogram, KeyType};
use trait_set::trait_set;

use histogram_macros::histogram_struct;
histogram_struct!{BTreeHistogram, BTreeHistKey, BTreeMap, BTreeSet, Iter, Ord}

trait_set! {
    pub trait LabelType = KeyType + Ord;
    pub trait FeatureType = Hash + Clone + Eq + PartialEq;
}

pub trait NaiveBayesExtractor {
    type InputValue;
    type Feature: FeatureType;

    fn extract(&self, value: &Self::InputValue) -> Vec<Self::Feature>;
}

pub struct NaiveBayes<L: LabelType, V, F: FeatureType, E:NaiveBayesExtractor<InputValue=V, Feature=F>> {
    extractor: E,
    label_counts: BTreeHistogram<L>,
    feature_counts: HashMap<F,HashHistogram<L>>
}

impl <L: LabelType, V, F: FeatureType, E:NaiveBayesExtractor<InputValue=V, Feature=F>> NaiveBayes<L,V,F,E> {
    pub fn new(extractor: E) -> Self {
        Self { extractor, label_counts: BTreeHistogram::new(), feature_counts: HashMap::new()}
    }

    pub fn p_label(&self, label: &L) -> f64 {
        self.label_counts.count(label) as f64 / self.label_counts.len() as f64
    }
}

impl <L: LabelType, V, F: FeatureType, E:NaiveBayesExtractor<InputValue=V, Feature=F>> Classifier<V,L> for NaiveBayes<L,V,F,E> {
    fn train(&mut self, training_images: &Vec<(L,V)>) {
        for (label, value) in training_images.iter() {
            self.label_counts.bump(label);
            for feature in self.extractor.extract(value) {
                if !self.feature_counts.contains_key(&feature) {
                    self.feature_counts.insert(feature.clone(), HashHistogram::new());
                }
                self.feature_counts.get_mut(&feature).unwrap().bump(label);
            }            
        }
    }

    fn classify(&self, example: &V) -> L {
        let mut label_probs = self.label_counts.iter().map(|(label,_)| (label, 1.0)).collect::<BTreeMap<_,_>>();
        for feature in self.extractor.extract(example) {
            for (label, label_total) in self.label_counts.iter() {
                let label_total = *label_total + 1;
                if let Some(fcounts) = self.feature_counts.get(&feature) {
                    let count = fcounts.count(label) + 1;
                    (*label_probs.get_mut(label).unwrap()) *= count as f64 / label_total as f64 * self.p_label(label);
                }
            }
        }

        let mut rankings = label_probs.iter().map(|(label, prob)| (*prob, (*label).clone())).collect::<Vec<_>>();
        rankings.sort_by(cmp_w_label);
        println!("{rankings:?}");
        rankings.last().unwrap().1.clone()
    }
}

fn cmp_w_label<L: LabelType, V: Copy + PartialEq + PartialOrd>(a: &(V, L), b: &(V, L)) -> Ordering {
    cmp_f64(&a.0, &b.0)
}

// Borrowed from: https://users.rust-lang.org/t/sorting-vector-of-vectors-of-f64/16264
fn cmp_f64<M: Copy + PartialEq + PartialOrd>(a: &M, b: &M) -> Ordering {
    if a < b {
        return Ordering::Less;
    } else if a > b {
        return Ordering::Greater;
    }
    return Ordering::Equal;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Extractor {}

    impl NaiveBayesExtractor for Extractor {
        type InputValue = Vec<(char, i32)>;

        type Feature = (char, i32);

        fn extract(&self, value: &Self::InputValue) -> Vec<Self::Feature> {
            value.clone()
        }
    }

    #[test]
    fn it_works() {
        // P(A) = 3/5, P(B) = 2/5
        // Add 1 to numerator and denominator each time to prevent zeros
        //
        // P('A' | ('X', 5)) = P(('X', 5) | 'A') P('A') = (2/3) 3/4 * 3/5 = 9/20
        // P('B' | ('X', 5)) = P(('X', 5) | 'B') P('B') = (1/2) 2/3 * 2/5 = 4/15
        // P('A' | ('X', 3)) = P(('X', 3) | 'A') P('A') = (1/3) 2/4 * 3/5 = 6/20
        // P('B' | ('X', 3)) = P(('X', 3) | 'B') P('B') = (0/2) 1/3 * 2/5 = 2/15
        // P('A' | ('X', 4)) = P(('X', 4) | 'A') P('A') = (0/3) 1/4 * 3/5 = 3/20
        // P('B' | ('X', 4)) = P(('X', 4) | 'B') P('B') = (1/2) 2/3 * 2/5 = 4/15
        // P('A' | ('Y', 4)) = P(('Y', 4) | 'A') P('A') = (1/3) 2/4 * 3/5 = 6/20
        // P('B' | ('Y', 4)) = P(('Y', 4) | 'B') P('B') = (1/2) 2/3 * 2/5 = 4/15
        // P('A' | ('Y', 3)) = P(('Y', 3) | 'A') P('A') = (0/3) 1/4 * 3/5 = 3/20
        // P('B' | ('Y', 3)) = P(('Y', 3) | 'B') P('B') = (1/2) 2/3 * 2/5 = 4/15
        // P('A' | ('Y', 2)) = P(('Y', 2) | 'A') P('A') = (2/3) 3/4 * 3/5 = 9/20
        // P('B' | ('Y', 2)) = P(('Y', 2) | 'B') P('B') = (0/2) 1/3 * 2/5 = 2/15
        let training = vec![
            ('A', vec![('X', 5), ('Y', 4)]), 
            ('A', vec![('X', 5), ('Y', 2)]), 
            ('A', vec![('X', 3), ('Y', 2)]), 
            ('B', vec![('X', 4), ('Y', 4)]),
            ('B', vec![('X', 5), ('Y', 3)]),
        ];

        let testing = vec![
            // P('A') = 9/20 * 9/20, P('B') = 4/15 * 2/15
            ('A', vec![('X', 5), ('Y', 2)]),

            // P('A') = 3/20 * 9/20, P('B') = 4/15 * 2/15
            ('A', vec![('X', 4), ('Y', 2)]),

            // P('A') = 3/20, P('B') = 4/15
            ('B', vec![('X', 4), ('Y', 1)]),

            // P('A') = 9/20, P('B') = 4/15
            ('A', vec![('X', 5), ('Y', 1)]),

            // P('A') = 3/20, P('B') = 4/15
            ('B', vec![('X', 2), ('Y', 3)]),

            // P('A') = 6/20 * 3/20, P('B') = 2/15 * 4/15
            ('A', vec![('X', 3), ('Y', 3)]),
        ];

        let mut nb = NaiveBayes::new(Extractor {});
        nb.train(&training);

        for (label, example) in testing.iter() {
            let result = nb.classify(example);
            println!("{label} {example:?} ? {result}");
            assert_eq!(result, *label);
        }
    }
}
