//! Progress bars for `cargo run --bin movegen-o1-gen` (real terminal, indicatif).

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::time::Duration;

pub struct PhaseBar {
    pb: ProgressBar,
}

impl PhaseBar {
    pub fn new() -> Self {
        let pb = ProgressBar::new(1);
        pb.set_draw_target(ProgressDrawTarget::stderr_with_hz(20));
        pb.set_style(
            ProgressStyle::with_template(
                "{msg:.bold.cyan}\n  {spinner:.green} [{bar:48.cyan/blue}] {percent:>3}% {pos}/{len} {elapsed_precise}",
            )
            .expect("progress template")
            .progress_chars("█▉▊▋▌▍▎▏  "),
        );
        pb.enable_steady_tick(Duration::from_millis(80));
        Self { pb }
    }

    pub fn begin(&self, total: u64, title: &str) {
        let total = total.max(1);
        self.pb.set_length(total);
        self.pb.set_position(0);
        self.pb.set_message(title.to_string());
        self.pb.reset_elapsed();
    }

    pub fn tick(&self, detail: &str) {
        self.pb.inc(1);
        self.pb.set_message(detail.to_string());
    }

    pub fn finish(&self, detail: &str) {
        self.pb.finish_with_message(format!("✓ {detail}"));
    }
}

pub fn phase_bar() -> PhaseBar {
    PhaseBar::new()
}
