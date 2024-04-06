use opencv::{core::*, highgui::*, imgproc::*, types::VectorOfMat, videoio::*, Result};

fn main() -> Result<()> {
    let mut cap = VideoCapture::new(0, CAP_ANY)?;
    assert!(cap.is_opened()?);

    let mut src = Mat::default();
    let mut gray = Mat::default();
    let mut eroded = Mat::default();
    let mut thr_mask = Mat::default();
    let mut thr_mask_bgr = Mat::default();
    let mut bwa_thr_mask = Mat::default();

    named_window("Src", 0)?;
    create_trackbar("Threshold", "Src", None, 255, None)?;

    loop {
        let ticker = get_tick_count()?;
        assert!(cap.read(&mut src).unwrap()); // get frame to src mut Mat

        cvt_color(&src, &mut gray, COLOR_BGR2GRAY, 0)?;
        blur(
            &gray.clone(),
            &mut gray,
            Size::new(3, 3),
            Point::new(-1, -1),
            BORDER_DEFAULT,
        )?;

        let thr_val = get_trackbar_pos("Threshold", "Src")?;
        threshold(&gray, &mut thr_mask, thr_val as f64, 255., THRESH_BINARY)?;

        erode(
            &thr_mask,
            &mut eroded,
            &Mat::default(),
            Point::new(-1, -1),
            2,
            BORDER_DEFAULT,
            Scalar::all(1.),
        )?;
        dilate(
            &eroded,
            &mut thr_mask,
            &Mat::default(),
            Point::new(-1, -1),
            2,
            BORDER_DEFAULT,
            Scalar::all(1.),
        )?;

        cvt_color(&thr_mask, &mut thr_mask_bgr, COLOR_GRAY2BGR, 0)?;
        bitwise_and(&src, &thr_mask_bgr, &mut bwa_thr_mask, &Mat::default())?;

        let mut conts = VectorOfMat::default();
        find_contours(
            &thr_mask,
            &mut conts,
            RETR_EXTERNAL,
            CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        )?;

        let mut max_area = 0. as f64;
        let mut max_area_cont = Mat::default();
        for cont in conts.iter() {
            let area = contour_area(&cont, false)?;
            if area > max_area {
                max_area = area;
                max_area_cont = cont
            }
        }

        let rect = bounding_rect(&max_area_cont)?;
        rectangle(
            &mut thr_mask_bgr,
            rect,
            VecN([255., 0., 255., 0.]),
            1,
            LINE_AA,
            0,
        )?;
        circle(
            &mut thr_mask_bgr,
            Point::new(rect.x + rect.width / 2, rect.y + rect.height / 2),
            3,
            VecN([255., 0., 255., 0.]),
            -1,
            LINE_AA,
            0,
        )?;

        let moments = moments(&max_area_cont, false)?;
        circle(
            &mut thr_mask_bgr,
            Point::new(
                (moments.m10 / moments.m00) as i32,
                (moments.m01 / moments.m00) as i32,
            ),
            3,
            VecN([100., 255., 100., 0.]),
            -1,
            LINE_AA,
            0,
        )?;

        let fps = (1. / ((get_tick_count()? - ticker) as f64 / get_tick_frequency()?)) as i32;
        put_text(
            &mut thr_mask_bgr, 
            &fps.to_string(),
            Point::new(10, 30), 
            FONT_HERSHEY_COMPLEX_SMALL, 
            1., 
            VecN([255., 0., 255., 0.]), 
            1,
            LINE_AA, 
            false,
        )?;

        imshow("Src", &thr_mask_bgr)?;

        match wait_key(1)? {
            113 => break,
            27 => break,
            _ => continue,
        }
    }

    cap.release()?;
    Ok(())
}
