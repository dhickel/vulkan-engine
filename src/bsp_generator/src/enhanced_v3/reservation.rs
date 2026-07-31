//! Protected volume reservations for Enhanced V3.
//!
//! Manages immutable reservation volumes for routes, portals, spawn points,
//! and lights. Reservations are validated against each other for conflicts
//! and against assembly brushes for intrusion.

use super::assembly::ProtectedVolume;
use super::config::CONSTRUCTION_QUANTUM;
use super::error::V3Error;
use super::geometry::ConvexBrush;
use super::ids::QuantumVolume;

// ── Reservation record ─────────────────────────────────────────────────────

/// A single protected volume reservation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reservation {
    /// Stable reservation ID.
    pub id: String,
    /// Human-readable kind tag.
    pub kind: String,
    /// The quantum-aligned volume.
    pub volume: QuantumVolume,
}

impl Reservation {
    pub fn new(id: impl Into<String>, kind: impl Into<String>, volume: QuantumVolume) -> Self {
        Self {
            id: id.into(),
            kind: kind.into(),
            volume,
        }
    }

    /// Convert to an assembly ProtectedVolume.
    pub fn to_protected_volume(&self) -> Result<ProtectedVolume, V3Error> {
        let _quantum = CONSTRUCTION_QUANTUM as i128;
        let brush = ConvexBrush::make_box(
            (self.volume.x0 as i128, self.volume.x1 as i128),
            (self.volume.y0 as i128, self.volume.y1 as i128),
            (self.volume.z0 as i128, self.volume.z1 as i128),
        )?;
        Ok(ProtectedVolume {
            id: self.id.clone(),
            brush,
        })
    }
}

// ── Reservation set ────────────────────────────────────────────────────────

/// A validated set of protected volume reservations.
#[derive(Debug, Clone)]
pub struct ReservationSet {
    reservations: Vec<Reservation>,
}

impl ReservationSet {
    pub fn new() -> Self {
        Self {
            reservations: Vec::new(),
        }
    }

    /// Add a reservation, checking for conflicts with existing ones.
    pub fn add(&mut self, reservation: Reservation) -> Result<(), V3Error> {
        for existing in &self.reservations {
            if reservation.volume.intersects(&existing.volume) {
                return Err(V3Error::ReservationConflict {
                    resource: reservation.id.clone(),
                    existing: existing.id.clone(),
                });
            }
        }
        self.reservations.push(reservation);
        Ok(())
    }

    /// All reservations.
    pub fn all(&self) -> &[Reservation] {
        &self.reservations
    }

    /// Convert all reservations to assembly protected volumes.
    pub fn to_protected_volumes(&self) -> Result<Vec<ProtectedVolume>, V3Error> {
        self.reservations
            .iter()
            .map(|r| r.to_protected_volume())
            .collect()
    }

    /// Check that no two reservations overlap.
    pub fn validate_no_overlaps(&self) -> Result<(), V3Error> {
        for i in 0..self.reservations.len() {
            for j in (i + 1)..self.reservations.len() {
                if self.reservations[i]
                    .volume
                    .intersects(&self.reservations[j].volume)
                {
                    return Err(V3Error::ReservationConflict {
                        resource: self.reservations[i].id.clone(),
                        existing: self.reservations[j].id.clone(),
                    });
                }
            }
        }
        Ok(())
    }
}

// ── Reservation builder ────────────────────────────────────────────────────

/// Build the standard reservation set from topology output.
pub fn build_reservations(
    spawn_volume: QuantumVolume,
    light_volumes: &[QuantumVolume],
    route_envelope: Option<(i32, i32, i32, i32)>,
    transition_volume: Option<(i32, i32, i32, i32, i32, i32)>,
) -> Result<ReservationSet, V3Error> {
    let mut set = ReservationSet::new();

    // Spawn reservation
    set.add(Reservation::new("spawn", "spawn_point", spawn_volume))?;

    // Light reservations
    for (i, vol) in light_volumes.iter().enumerate() {
        set.add(Reservation::new(format!("light_{i:04}"), "light", *vol))?;
    }

    // Route envelope reservation
    if let Some((x0, y0, x1, y1)) = route_envelope {
        if let Some(vol) = QuantumVolume::new(x0, y0, 0, x1, y1, 384) {
            set.add(Reservation::new("route_primary", "route", vol))?;
        }
    }

    // Transition protected volume
    if let Some((x0, y0, z0, x1, y1, z1)) = transition_volume {
        if let Some(vol) = QuantumVolume::new(x0, y0, z0, x1, y1, z1) {
            set.add(Reservation::new("transition_primary", "transition", vol))?;
        }
    }

    set.validate_no_overlaps()?;

    Ok(set)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::config::CONSTRUCTION_QUANTUM;
    use super::*;

    #[test]
    fn empty_set_validates() {
        let set = ReservationSet::new();
        assert!(set.validate_no_overlaps().is_ok());
        assert!(set.all().is_empty());
    }

    #[test]
    fn disjoint_reservations_pass() {
        let q = CONSTRUCTION_QUANTUM;
        let mut set = ReservationSet::new();
        set.add(Reservation::new(
            "a",
            "spawn",
            QuantumVolume::new(0, 0, 0, q, q, q).unwrap(),
        ))
        .unwrap();
        set.add(Reservation::new(
            "b",
            "light",
            QuantumVolume::new(q, q, q, 2 * q, 2 * q, 2 * q).unwrap(),
        ))
        .unwrap();
        assert!(set.validate_no_overlaps().is_ok());
    }

    #[test]
    fn overlapping_reservations_rejected() {
        let q = CONSTRUCTION_QUANTUM;
        let mut set = ReservationSet::new();
        set.add(Reservation::new(
            "a",
            "spawn",
            QuantumVolume::new(0, 0, 0, 2 * q, 2 * q, 2 * q).unwrap(),
        ))
        .unwrap();
        let result = set.add(Reservation::new(
            "b",
            "light",
            QuantumVolume::new(q, q, q, 3 * q, 3 * q, 3 * q).unwrap(),
        ));
        assert!(result.is_err());
    }

    #[test]
    fn build_standard_reservations() {
        let q = CONSTRUCTION_QUANTUM;
        let spawn = QuantumVolume::new(0, 0, 0, q, q, q).unwrap();
        let lights = vec![QuantumVolume::new(2 * q, 2 * q, 2 * q, 3 * q, 3 * q, 3 * q).unwrap()];

        let set = build_reservations(spawn, &lights, None, None).unwrap();
        assert_eq!(set.all().len(), 2); // spawn + 1 light
    }

    #[test]
    fn reservation_to_protected_volume() {
        let q = CONSTRUCTION_QUANTUM;
        let res = Reservation::new(
            "test_pv",
            "spawn",
            QuantumVolume::new(0, 0, 0, q, q, q).unwrap(),
        );
        let pv = res.to_protected_volume().unwrap();
        assert_eq!(pv.id, "test_pv");
        assert!(pv.brush.volume() > super::super::geometry::Rational::ZERO);
    }
}
