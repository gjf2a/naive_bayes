use std::{sync::Arc, hash::Hash, collections::{HashMap, BTreeSet, BTreeMap, btree_map::Iter}};

use supervised_learning::Classifier;
use hash_histogram::{HashHistogram, KeyType};
use trait_set::trait_set;

use histogram_macros::histogram_struct;
histogram_struct!{BTreeHistogram, BTreeHistKey, BTreeMap, BTreeSet, Iter, Ord}

trait_set! {
    pub trait LabelType = KeyType + Ord;
    pub trait FeatureType = Hash + Clone + Eq + PartialEq;
}

pub struct NaiveBayes<L: LabelType, V, F: FeatureType> {
    extractor: Arc<fn(&V)->Vec<F>>,
    label_counts: BTreeHistogram<L>,
    feature_counts: HashMap<F,HashHistogram<L>>
}

impl <L: LabelType, V, F: FeatureType> NaiveBayes<L,V,F> {
    pub fn new(extractor: Arc<fn(&V)->Vec<F>>) -> Self {
        Self { extractor, label_counts: BTreeHistogram::new(), feature_counts: HashMap::new()}
    }

    pub fn p_label(&self, label: &L) -> f64 {
        self.label_counts.count(label) as f64 / self.label_counts.len() as f64
    }
}

impl <L: LabelType, V, F: FeatureType> Classifier<V,L> for NaiveBayes<L,V,F> {
    fn train(&mut self, training_images: &Vec<(L,V)>) {
        for (label, value) in training_images.iter() {
            self.label_counts.bump(label);
            for feature in (self.extractor)(value) {
                if !self.feature_counts.contains_key(&feature) {
                    self.feature_counts.insert(feature.clone(), HashHistogram::new());
                }
                self.feature_counts.get_mut(&feature).unwrap().bump(label);
            }            
        }
    }

    fn classify(&self, example: &V) -> L {
        let mut counts = BTreeHistogram::new();
        for feature in (self.extractor)(example) {
            if let Some(fcounts) = self.feature_counts.get(&feature) {
                for label in fcounts.iter() {
                    counts.bump_by(*label, fcounts.count(*label));
                }
            }
        }
        
    }
}

// Pregenerated code
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
