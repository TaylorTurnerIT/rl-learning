use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use dodge_batch::{
    BatchConfig, BatchEnvironment, BatchObservation, ExecutionMode, ObservationFlags,
};
use dodge_core::Action;

const LANE_COUNT: usize = 32;
const BATCH_STEPS: usize = 32;

fn full_observation_batch(c: &mut Criterion) {
    c.bench_function("batch/full_state_pixels", |bencher| {
        bencher.iter(|| {
            let mut config = BatchConfig::new(4);
            config.execution = ExecutionMode::Parallel;
            config.observations = ObservationFlags::pixels_and_state();
            let Ok(mut environment) = BatchEnvironment::new(config) else {
                return;
            };

            let seeds: Vec<u32> = (0..LANE_COUNT)
                .map(|lane| 42_u32.saturating_add((lane as u32).saturating_mul(97)))
                .collect();
            let Ok(initial) = environment.reset(&seeds) else {
                return;
            };
            let mut digest = digest_observations(&initial);
            let mut actions = vec![Action::Neutral; LANE_COUNT];
            for step in 0..BATCH_STEPS {
                for (lane, action) in actions.iter_mut().enumerate() {
                    let action_index = (step * LANE_COUNT + lane) % Action::ALL.len();
                    if let Some(next) = Action::ALL.get(action_index).copied() {
                        *action = next;
                    }
                }
                let Ok(observations) = environment.step(&actions) else {
                    return;
                };
                digest ^= digest_observations(&observations);
            }
            black_box(digest);
        });
    });
}

fn digest_observations(observations: &[BatchObservation]) -> u64 {
    observations.iter().fold(0_u64, |digest, observation| {
        let pixels = observation.pixels.as_ref().map_or(0_u64, |value| {
            value
                .iter()
                .fold(0_u64, |sum, pixel| sum + u64::from(*pixel))
        });
        digest
            ^ u64::from(observation.frame)
            ^ observation.state_hash
            ^ observation.pixel_hash
            ^ pixels
    })
}

criterion_group!(benches, full_observation_batch);
criterion_main!(benches);
