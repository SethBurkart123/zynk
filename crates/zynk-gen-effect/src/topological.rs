//! Stable topological ordering for model emission.

use std::collections::{BTreeMap, BTreeSet};

/// Order models so dependencies are emitted before dependents.
///
/// Uses Kahn's algorithm with alphabetical tie-breaks. Self-references should
/// already be removed from each dependency set by the caller. If a cycle
/// remains, the cyclic tail is appended alphabetically to preserve deterministic
/// output while matching the Python generator's fallback behavior.
pub fn sort_models(deps_by_model: &BTreeMap<String, BTreeSet<String>>) -> Vec<String> {
    let mut in_degree: BTreeMap<String, usize> =
        deps_by_model.keys().map(|name| (name.clone(), 0)).collect();
    let mut reverse: BTreeMap<String, BTreeSet<String>> = deps_by_model
        .keys()
        .map(|name| (name.clone(), BTreeSet::new()))
        .collect();

    for (name, deps) in deps_by_model {
        for dep in deps {
            if !deps_by_model.contains_key(dep) {
                continue;
            }
            *in_degree.entry(name.clone()).or_default() += 1;
            reverse.entry(dep.clone()).or_default().insert(name.clone());
        }
    }

    let mut ready: Vec<String> = in_degree
        .iter()
        .filter_map(|(name, degree)| (*degree == 0).then_some(name.clone()))
        .collect();
    ready.sort();

    let mut ordered = Vec::new();
    while !ready.is_empty() {
        let name = ready.remove(0);
        ordered.push(name.clone());

        for dependent in reverse.get(&name).into_iter().flatten() {
            let Some(degree) = in_degree.get_mut(dependent) else {
                continue;
            };
            *degree -= 1;
            if *degree == 0 {
                ready.push(dependent.clone());
            }
        }
        ready.sort();
    }

    if ordered.len() < deps_by_model.len() {
        let emitted: BTreeSet<_> = ordered.iter().cloned().collect();
        ordered.extend(
            deps_by_model
                .keys()
                .filter(|name| !emitted.contains(*name))
                .cloned(),
        );
    }

    ordered
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    fn set(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn orders_dependencies_before_dependents_with_alphabetical_ties() {
        let deps = BTreeMap::from([
            ("Z".to_string(), set(&["A", "M"])),
            ("M".to_string(), set(&["A"])),
            ("A".to_string(), BTreeSet::new()),
            ("B".to_string(), BTreeSet::new()),
        ]);

        assert_eq!(super::sort_models(&deps), vec!["A", "B", "M", "Z"]);
    }

    #[test]
    fn appends_cyclic_remainder_alphabetically() {
        let deps = BTreeMap::from([
            ("C".to_string(), set(&["B"])),
            ("B".to_string(), set(&["C"])),
            ("A".to_string(), BTreeSet::new()),
        ]);

        assert_eq!(super::sort_models(&deps), vec!["A", "B", "C"]);
    }
}
