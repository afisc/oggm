"""Dynamic spinup functions for model initialisation using the IGM core"""
# Builtins
import logging
import copy
import os
import warnings

# External libs
import numpy as np
from scipy import interpolate
import xarray as xr
# Locals
import oggm.cfg as cfg
from oggm import utils

# workflow and gcm_climate import used for tests with GCM climate
from oggm import workflow
from oggm.shop import gcm_climate

from oggm import entity_task
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.core.massbalance import (MultipleFlowlineMassBalance,
                                   ConstantMassBalance,
                                   DistributedMassBalance,
                                   MonthlyTIModel,
                                   apparent_mb_from_any_mb)
from oggm.core.sia2d import IGM_Model2D
from oggm.shop import cook23
# Module logger
log = logging.getLogger(__name__)

@entity_task(log)
def run_dynamic_ioggm_spinup(gdir, init_model_filesuffix=None, init_model_yr=None,
                       init_model_geom=None,
                       climate_input_filesuffix='',
                       use_gcm_climate= False,
                       evolution_model=None,
                       mb_model_historical=None, mb_model_spinup=None,
                       spinup_period=20, spinup_start_yr=None,
                       min_spinup_period=10, spinup_start_yr_max=None,
                       target_yr=None, target_value=None,
                       minimise_for='area', precision_percent=1,
                       precision_absolute=1, min_ice_thickness=None,
                       first_guess_t_spinup=-2, t_spinup_max_step_length=2,
                       maxiter=30, output_filesuffix='_dynamic_ioggm_spinup',
                       store_model_geometry=True, store_model_geometry_spinup=False,
                       store_diagnostics=None, store_diagnostics_spinup=False,
                       store_model_evolution=True, ignore_errors=False,
                       return_t_spinup_best=False, ye=None,
                       model_flowline_filesuffix='',
                       add_fixed_geometry_spinup=False, allow_calving=False,
                       store_monthly_step=None,
                       store_all_spinup_steps=False,
                             **kwargs):
    """
    TODO: adapt docstring to ioggm_spinup
    Dynamically spinup the glacier to match area or volume at the RGI date.

        This task allows to do simulations in the recent past (before the glacier
        inventory date), when the state of the glacier was unknown. This is a
        very difficult task the longer further back in time one goes
        (see publications by Eis et al. for a theoretical background), but can
        work quite well for short periods. Note that the solution is not unique.

        Parameters
        ----------
        gdir : :py:class:`oggm.GlacierDirectory`
            the glacier directory to process
        init_model_filesuffix : str or None
            if you want to start from a previous model run state. This state
            should be at time yr_rgi_date.
        init_model_yr : int or None
            the year of the initial run you want to start from. The default
            is to take the last year of the simulation.
        init_model_geom : xarray.DataArray
            2D xarray data to initialise the model (the default is the
            cook_23 dataset).
            Ignored if `init_model_filesuffix` is set
        climate_input_filesuffix : str
            filesuffix for the input climate file
        evolution_model : :class:oggm.core.FlowlineModel
            which evolution model to use. Default: cfg.PARAMS['evolution_model']
            Not all models work in all circumstances!
        mb_model_historical : :py:class:`core.MassBalanceModel`
            User-povided MassBalanceModel instance for the historical run. Default
            is to use a MonthlyTIModel model  together with the provided
            parameter climate_input_filesuffix.
        mb_model_spinup : :py:class:`core.MassBalanceModel`
            User-povided MassBalanceModel instance for the spinup before the
            historical run. Default is to use a ConstantMassBalance model together
            with the provided parameter climate_input_filesuffix and during the
            period of spinup_start_yr until rgi_year (e.g. 1979 - 2000).
        spinup_period : int
            The period how long the spinup should run. Start date of historical run
            is defined "target_yr - spinup_period". Minimum allowed value is 10. If
            the provided climate data starts at year later than
            (target_yr - spinup_period) the spinup_period is set to
            (target_yr - yr_climate_start). Caution if spinup_start_yr is set the
            spinup_period is ignored.
            Default is 20
        spinup_start_yr : int or None
            The start year of the dynamic spinup. If the provided year is before
            the provided climate data starts the year of the climate data is used.
            If set it overrides the spinup_period.
            Default is None
        min_spinup_period : int
            If the dynamic spinup function fails with the initial 'spinup_period'
            a shorter period is tried. Here you can define the minimum period to
            try.
            Default is 10
        spinup_start_yr_max : int or None
            Possibility to provide a maximum year where the dynamic spinup must
            start from at least. If set, this overrides the min_spinup_period if
            target_yr - spinup_start_yr_max > min_spinup_period.
            Default is None
        target_yr : int or None
            The year at which we want to match area or volume.
            If None, gdir.rgi_date + 1 is used (the default).
            Default is None
        target_value : float or None
            The value we want to match at target_yr. Depending on minimise_for this
            value is interpreted as an area in km2 or a volume in km3. If None the
            total area or volume from the provided initial flowlines is used.
            Default is None
        minimise_for : str
            The variable we want to match at target_yr. Options are 'area' or
            'volume'.
            Default is 'area'
        precision_percent : float
            Gives the precision we want to match in percent. The algorithm makes
            sure that the resulting relative mismatch is smaller than
            precision_percent, but also that the absolute value is smaller than
            precision_absolute.
            Default is 1., meaning the difference must be within 1% of the given
            value (area or volume).
        precision_absolute : float
            Gives an minimum absolute value to match. The algorithm makes sure that
            the resulting relative mismatch is smaller than precision_percent, but
            also that the absolute value is smaller than precision_absolute.
            The unit of precision_absolute depends on minimise_for (if 'area' in
            km2, if 'volume' in km3)
            Default is 1.
        min_ice_thickness : float
            Gives an minimum ice thickness for model grid points which are counted
            to the total model value. This could be useful to filter out seasonal
            'glacier growth', as OGGM do not differentiate between snow and ice in
            the forward model run. Therefore you could see quite fast changes
            (spikes) in the time-evolution (especially visible in length and area).
            If you set this value to 0 the filtering can be switched off.
            Default is cfg.PARAMS['dynamic_spinup_min_ice_thick'].
        first_guess_t_spinup : float
            The initial guess for the temperature bias for the spinup
            MassBalanceModel in °C.
            Default is -2.
        t_spinup_max_step_length : float
            Defines the maximums allowed change of t_spinup between two iterations.
            Is needed to avoid to large changes.
            Default is 2.
        maxiter : int
            Maximum number of minimisation iterations per spinup period. If reached
            and 'ignore_errors=False' an error is raised.
            Default is 30
        output_filesuffix : str
            for the output file
        store_model_geometry : bool
            whether to store the full model geometry run file to disk or not.
            Default is True
        store_all_spinup_steps : bool or None
            whether to store the diagnostics of every spinup iteration or not.
            Default is None
        store_model_evolution : bool
            if True the complete dynamic spinup run is saved (complete evolution
            of the model during the dynamic spinup), if False only the final model
            state after the dynamic spinup run is saved. (Hint: if
            store_model_evolution = True and ignore_errors = True and an Error
            during the dynamic spinup occurs the stored model evolution is only one
            year long)
            Default is True
        ignore_errors : bool
            If True the function saves the model without a dynamic spinup using
            the 'output_filesuffix', if an error during the dynamic spinup occurs.
            This is useful if you want to keep glaciers for the following tasks.
            Default is True
        return_t_spinup_best : bool
            If True the used temperature bias for the spinup is returned in
            addition to the final model. If an error occurs and ignore_error=True,
            the returned value is np.nan.
            Default is False
        ye : int
            end year of the model run, must be larger than target_yr. If nothing is
            given it is set to target_yr. It is not recommended to use it if only
            data until target_yr is needed for calibration as this increases the
            run time of each iteration during the iterative minimisation. Instead
            use run_from_climate_data afterwards and merge both outputs using
            merge_consecutive_run_outputs.
            Default is None
        model_flowline_filesuffix : str
            suffix to the model_flowlines filename to use (if no other flowlines
            are provided with init_model_filesuffix or init_model_fls).
            Default is ''
        add_fixed_geometry_spinup : bool
            If True and the original spinup_period must be shortened (due to
            ice-free or out-of-boundary error) a fixed geometry spinup is added at
            the beginning so that the resulting model run always starts from the
            defined start year (could be defined through spinup_period or
            spinup_start_yr). Only has an effect if store_model_evolution is True.
            Default is False
        allow_calving : bool
            If True you can use the dynamic spinup with calving. So far this is not
            tested and you need to know what you are doing when switching it on.
            Default is False
        store_monthly_step : Bool
            If True (False)  model diagnostics will be stored monthly (yearly).
            If unspecified, we follow the update of the MB model, which
            defaults to yearly (see __init__).
        use_gcm_climate: Bool
            If True, a gcm climate will be loaded and used for the whole spinup.
            The parameter 'climate_input_filesuffix' will not be used.
            If False, the 'climate_historical' will be used.
        kwargs : dict
            kwargs to pass to the evolution_model instance

        Returns
        -------
        :py:class:`oggm.core.flowline.evolution_model`
            The final dynamically spined-up model. Type depends on the selected
            evolution_model.
        """

    # use the IGM_Model2D (default for iOGGM)
    evolution_model = IGM_Model2D

    # load the IGM Inversion as a starting ice volume
    cook23.cook23_to_gdir(gdir)

    # load the gridded data that's needed for the 2DModel
    with xr.open_dataset(gdir.get_filepath('gridded_data')) as gd:
        gd = gd.load()

    # set values outside the glacier to np.nan
    # using the glacier mask, as otherwise there is more ice from surrounding glaciers in the domain,
    # which shouldn't accumulate more ice, still adds to the total volume/area of the domain.. either mask it out beforehand or before doing plots.
    gd['cook23_thk_masked'] = xr.where(gd.glacier_mask, gd.cook23_thk, np.nan)

    # load the glacier bed topography.
    bed_con = gd.topo - gd.consensus_ice_thickness.fillna(0)
    bed_cook_masked = gd.topo - gd.cook23_thk_masked.fillna(0)
    bed_cook = gd.topo - gd.cook23_thk

    if use_gcm_climate:
        climate_filename = 'gcm_data'
        # load gcm climate
        member = 'mri-esm2-0_r1i1p1f1'
        ssp = 'ssp370'

        climate_input_filesuffix = f'_ISIMIP3b_{member}_{ssp}'
        # bias correct them
        workflow.execute_entity_task(gcm_climate.process_monthly_isimip_data, gdir,
                                     ssp=ssp,
                                     # gcm member -> you can choose another one
                                     member=member,
                                     # recognize the climate file for later
                                     output_filesuffix=climate_input_filesuffix
                                     )

    else:
        climate_filename = 'climate_historical'
    # area of one gridpoint
    gridpoint_area = gdir.grid.dx**2

    if target_yr is None:
        # Even in calendar dates, we prefer to set rgi_year in the next year
        # as the rgi is often from snow free images the year before (e.g. Aug)
        target_yr = gdir.rgi_date + 1

    if ye is None:
        ye = target_yr

    if ye < target_yr:
        raise RuntimeError(f'The provided end year (ye = {ye}) must be larger'
                           f'than the target year (target_yr = {target_yr}!')

    yr_min = gdir.get_climate_info()['baseline_yr_0']

    if min_ice_thickness is None:
        min_ice_thickness = cfg.PARAMS['dynamic_spinup_min_ice_thick']

    # check provided maximum start year here, and change min_spinup_period
    if spinup_start_yr_max is not None:
        if spinup_start_yr_max < yr_min:
            raise RuntimeError(f'The provided maximum start year (= '
                               f'{spinup_start_yr_max}) must be larger than '
                               f'the start year of the provided climate data '
                               f'(= {yr_min})!')
        if spinup_start_yr is not None:
            if spinup_start_yr_max < spinup_start_yr:
                raise RuntimeError(f'The provided start year (= '
                                   f'{spinup_start_yr}) must be smaller than '
                                   f'the maximum start year '
                                   f'{spinup_start_yr_max}!')
        if (target_yr - spinup_start_yr_max) > min_spinup_period:
            min_spinup_period = (target_yr - spinup_start_yr_max)


    if init_model_filesuffix is not None:
        fp = gdir.get_filepath('ioggm_geometry', filesuffix=init_model_filesuffix)
        init_model = xr.open_dataset(fp)
        if init_model_yr is None:
            # get the last model year
            init_model_yr = init_model.coords["time"].values[-1]
        init_model_geom = init_model.ice_thickness.sel(time=init_model_yr)



    if init_model_geom is None:
        # in the original spinup this would load a
        # 'copy' of the just done inversion(for the melt_f adaption). as we don't do an inversion we just always start
        # from cook data or an actual geometry that is passed.
        model_geom_spinup = gd['cook23_thk_masked']
    else:
        model_geom_spinup = copy.copy(init_model_geom)
    # TODO: maybe add a check if the data passed matches the resolution of the gdir
    # utils.model_geom_is_valid(gdir, model_geom_spinup)


        # MassBalance for actual run from yr_spinup to target_yr
    if mb_model_historical is None:
        mb_model_historical = DistributedMassBalance(
            gdir, mb_model_class=MonthlyTIModel,
            filename=climate_filename,
            input_filesuffix=climate_input_filesuffix,
            ### parameters have been used for experimental tests in the beginning.
            # temp_bias = 2,
            # melt_f = 10,
                )


    # here we define the file-paths for the output
    if store_model_geometry:
        geom_path = gdir.get_filepath('ioggm_geometry',
                                      filesuffix=output_filesuffix,
                                      delete=True)
    else:
        geom_path = False

    if store_model_geometry_spinup:
        geom_path_spinup = gdir.get_filepath('ioggm_geometry',
                                             filesuffix=output_filesuffix+'_spinup',
                                             delete=True)
    else:
        geom_path_spinup = False

    if store_diagnostics is None:
        store_diagnostics = cfg.PARAMS['store_fl_diagnostics']

    if store_diagnostics:
        ioggm_diag_path = gdir.get_filepath('ioggm_diagnostics',
                                         filesuffix=output_filesuffix,
                                         delete=True)
    else:
        ioggm_diag_path = False

    if store_diagnostics_spinup:
        ioggm_diag_path_spinup = gdir.get_filepath('ioggm_diagnostics',
                                                   filesuffix=output_filesuffix + '_spinup',
                                                   delete=True)
    else:
        ioggm_diag_path_spinup = False

    diag_path = gdir.get_filepath('model_diagnostics',
                                  filesuffix=output_filesuffix,
                                  delete=True)

    ### LEAVING OUT the 'use_inversion_params_for_run' OPTION. could be added later.
    ### -----
    ### -----
    ### -----

    fs = cfg.PARAMS['fs']
    glen_a = cfg.PARAMS['glen_a']
    # kwargs.setdefault('fs', fs) # not passing the fs parameter as it is a unknown parameter to the 2DModel
    kwargs.setdefault('glen_a', glen_a)

    mb_elev_feedback = kwargs.get('mb_elev_feedback', 'annual')
    if mb_elev_feedback != 'annual':
        raise InvalidParamsError('Only use annual mb_elev_feedback with the '
                                 'dynamic spinup function!')

    ### LEAVING OUT the 'use_kcalving_for_run' OPTION. could be added later.
    ### -----
    ### -----
    ### -----


    def save_model_without_dynamic_spinup():
        gdir.add_to_diagnostics('run_dynamic_spinup_success', False)
        yr_use = np.clip(target_yr, yr_min, None)
        model_dynamic_spinup_end = evolution_model(bed_con.values,
                                                   init_ice_thick=model_geom_spinup.fillna(0).values,
                                                   dx=gdir.grid.dx, dy=gdir.grid.dy, x=bed_con.x, y=bed_con.y,
                                                   mb_model=mb_model_historical,
                                                   y0=yr_use, mb_filter=gd.glacier_mask.values == 1)

        with warnings.catch_warnings():
            if ye < yr_use:
                yr_run = yr_use
            else:
                yr_run = ye
            # For operational runs we ignore the warnings
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            model_dynamic_spinup_end.run_until_and_store(
                yr_run,
                geom_path=geom_path,
                # diag_path=diag_path,
                # fl_diag_path=fl_diag_path,
                grid=gdir.grid,
                # store_monthly_step=store_monthly_step,
                diag_path=ioggm_diag_path,
            )

        return model_dynamic_spinup_end

    # apply the above function if spinup can not be executed
    if target_yr < yr_min + min_spinup_period:
        log.warning('The provided rgi_date is smaller than yr_climate_start + '
                    'min_spinup_period, therefore no dynamic spinup is '
                    'conducted and the original flowlines are saved at the '
                    'provided target year or the start year of the provided '
                    'climate data (if yr_climate_start > target_yr)')
        if ignore_errors:
            model_dynamic_spinup_end = save_model_without_dynamic_spinup()
            if return_t_spinup_best:
                return model_dynamic_spinup_end, np.nan
            else:
                return model_dynamic_spinup_end
        else:
            raise RuntimeError('The difference between the rgi_date and the '
                               'start year of the climate data is too small to '
                               'run a dynamic spinup!')


    # calculate the volume and area
    volume = model_geom_spinup.sum(dim=['x', 'y']) * gdir.grid.dx ** 2 * 1e-9
    area = (model_geom_spinup > 1).sum(dim=['x', 'y']) * gdir.grid.dx ** 2 * 1e-6

    if minimise_for == 'area':
        unit = 'km2'
        other_variable = 'volume'
        other_unit = 'km3'
        if target_value is None:
            reference_value = area
        else:
            reference_value = target_value
        other_reference_value = volume
    elif minimise_for == 'volume':
        unit = 'km3'
        other_variable = 'area'
        other_unit = 'km2'
        if target_value is None:
            reference_value = volume
        else:
            reference_value = target_value
        other_reference_value = area
    else:
        raise NotImplementedError

    # if reference value is zero no dynamic spinup is possible
    if reference_value == 0.:
        if ignore_errors:
            model_dynamic_spinup_end = save_model_without_dynamic_spinup()
            if return_t_spinup_best:
                return model_dynamic_spinup_end, np.nan
            else:
                return model_dynamic_spinup_end
        else:
            raise RuntimeError('The given reference value is Zero, no dynamic '
                               'spinup possible!')

    # here we adjust the used precision_percent to make sure the resulting
    # absolute mismatch is smaller than precision_absolute
    precision_percent = min(precision_percent,
                            precision_absolute / reference_value * 100)

    # only used to check performance of function
    forward_model_runs = [0]

    # the actual spinup run
    def run_model_with_spinup_to_target_year(t_spinup):
        forward_model_runs.append(forward_model_runs[-1] + 1)

        # with t_spinup the glacier state after spinup is changed between iterations
        mb_model_spinup.temp_bias = t_spinup
        # run the spinup
        model_spinup = evolution_model(bed_con.values,
                                                   init_ice_thick=model_geom_spinup.fillna(0).values,
                                                   dx=gdir.grid.dx, dy=gdir.grid.dy, x=bed_con.x, y=bed_con.y,
                                                   mb_model=mb_model_spinup,
                                                   y0=yr_spinup-(2*halfsize_spinup)+1, mb_filter=gd.glacier_mask.values == 1)
        # model_spinup.run_until(2 * halfsize_spinup)
        ds_spinup = model_spinup.run_until_and_store(yr_spinup+1,
                                         geom_path=geom_path_spinup,
                                         grid=gdir.grid,
                                         diag_path=ioggm_diag_path_spinup,
                                         )
        # save all the spinup steps before the historical run starts(default: 1980)
        if store_all_spinup_steps:
            spinup_step_ds = xr.Dataset(
                coords={'time': ds_spinup.time},
            )
            # calculate timeserieses of volume and area
            area_km2 = (ds_spinup.ice_thickness > 2).sum(dim=['x', 'y']) * (gdir.grid.dx ** 2) * 1e-6
            volume_km3 = ds_spinup.ice_thickness.sum(dim=['x', 'y']) * (gdir.grid.dx ** 2) * 1e-9

            spinup_step_ds['area_km2'] = area_km2
            spinup_step_ds['volume_km3'] = volume_km3
            step_path_spinup = ioggm_diag_path_spinup[:-3] + f'_it{forward_model_runs[-1]}.nc'
            if os.path.exists(step_path_spinup):
                os.remove(step_path_spinup)
            spinup_step_ds.to_netcdf(step_path_spinup)
        # if glacier is completely gone return information in ice-free
        ice_free = False
        if np.isclose(model_spinup.volume_km3, 0.):
            ice_free = True

        # Now conduct the actual model run to the rgi date

        model_historical = evolution_model(bed_con.values,
                                           init_ice_thick=model_spinup.ice_thick,
                                           dx=gdir.grid.dx, dy=gdir.grid.dy, x=bed_con.x, y=bed_con.y,
                                           mb_model=mb_model_historical,
                                           y0=yr_spinup, mb_filter=gd.glacier_mask.values == 1, **kwargs)
        if store_model_evolution:
            # check if we need to add the min_h variable (done inplace)
            delete_area_min_h = False
            ovars = cfg.PARAMS['store_diagnostic_variables']
            if 'area_min_h' not in ovars:
                ovars += ['area_min_h']
                delete_area_min_h = True

            ds = model_historical.run_until_and_store(
                ye, # if not set differently manually, usually equals target_yr
                geom_path=geom_path,
                grid=gdir.grid,
                diag_path=ioggm_diag_path,
                step=1, # save the yearly timestamp (function's default is: 2)
                # dynamic_spinup_min_ice_thick=min_ice_thickness, # TBI
                # fixed_geometry_spinup_yr=fixed_geometry_spinup_yr, # TBI
                # store_monthly_step=store_monthly_step, # TBI
            )

            # now we delete the min_h variable again if it was not
            # included before (inplace)
            if delete_area_min_h:
                ovars.remove('area_min_h')
            # Store the last model run, even if it is not the final one. To see the algorithms evolution.
            if store_all_spinup_steps:
                step_ds = xr.Dataset(
                    coords={'time': ds.time},
                )
                # calculate timeserieses of volume and area
                area_km2 = (ds.ice_thickness > 2).sum(dim=['x', 'y']) * (gdir.grid.dx ** 2) * 1e-6
                volume_km3 = ds.ice_thickness.sum(dim=['x', 'y']) * (gdir.grid.dx ** 2) * 1e-9

                step_ds['area_km2'] = area_km2
                step_ds['volume_km3'] = volume_km3
                step_path = ioggm_diag_path[:-3] +  f'_it{forward_model_runs[-1]}.nc'
                if os.path.exists(step_path):
                    os.remove(step_path)
                step_ds.to_netcdf(step_path)
            if type(ds) == tuple:
                ds = ds[0]
            model_area_km2 = (ds.ice_thickness.loc[target_yr] > 1).sum(dim=['x', 'y']) * gridpoint_area * 1e-6
            model_volume_km3 = ds.ice_thickness.sum(dim=['x', 'y']).loc[target_yr] * gridpoint_area * 1e-9
        else:
            # only run to rgi date and extract values
            model_historical.run_until(target_yr)

            model_area_km2 = (model_historical.ice_thick > 1).sum() * gridpoint_area * 1e-6
            model_volume_km3 = np.sum(model_historical.ice_thick) * gridpoint_area * 1e-9
            # afterwards finish the complete run
            model_historical.run_until(ye)

        if minimise_for == 'area':
            return model_area_km2, model_volume_km3, model_historical, ice_free
        elif minimise_for == 'volume':
            return model_volume_km3, model_area_km2, model_historical, ice_free
        else:
            raise NotImplementedError(f'{minimise_for}')

    def cost_fct(t_spinup, model_dynamic_spinup_end_loc, other_variable_mismatch_loc):
        # actual model run
        model_value, other_value, model_dynamic_spinup, ice_free = \
            run_model_with_spinup_to_target_year(t_spinup)

        # save the final model for later
        model_dynamic_spinup_end_loc.append(copy.copy(model_dynamic_spinup))

        # calculate the mismatch in percent
        cost = (model_value - reference_value) / reference_value * 100
        other_variable_mismatch_loc.append(
            (other_value - other_reference_value) / other_reference_value * 100)

        return cost, ice_free

    def init_cost_fct():
        model_dynamic_spinup_end_loc = []
        other_variable_mismatch_loc = []

        def c_fct(t_spinup):
            return cost_fct(t_spinup, model_dynamic_spinup_end_loc,
                            other_variable_mismatch_loc)

        return c_fct, model_dynamic_spinup_end_loc, other_variable_mismatch_loc

    def minimise_with_spline_fit(fct_to_minimise):
        # defines limits of t_spinup in accordance to maximal allowed change
        # between iterations
        t_spinup_limits = [first_guess_t_spinup - t_spinup_max_step_length,
                           first_guess_t_spinup + t_spinup_max_step_length]
        t_spinup_guess = []
        mismatch = []
        # this two variables indicate that the limits were already adapted to
        # avoid an ice_free or out_of_domain error
        was_ice_free = False
        was_out_of_domain = False
        was_errors = [was_out_of_domain, was_ice_free]

        def get_mismatch(t_spinup):
            t_spinup = copy.copy(t_spinup)
            # first check if the new t_spinup is in limits
            if t_spinup < t_spinup_limits[0]:
                # was the smaller limit already executed, if not first do this
                if t_spinup_limits[0] not in t_spinup_guess:
                    t_spinup = copy.copy(t_spinup_limits[0])
                else:
                    # smaller limit was already used, check if it was
                    # already newly defined with glacier exceeding domain
                    if was_errors[0]:
                        raise RuntimeError('Not able to minimise without '
                                           'exceeding the domain! Best '
                                           f'mismatch '
                                           f'{np.min(np.abs(mismatch))}%')
                    else:
                        # ok we set a new lower limit
                        t_spinup_limits[0] = (t_spinup_limits[0] -
                                              t_spinup_max_step_length)
            elif t_spinup > t_spinup_limits[1]:
                # was the larger limit already executed, if not first do this
                if t_spinup_limits[1] not in t_spinup_guess:
                    t_spinup = copy.copy(t_spinup_limits[1])
                else:
                    # larger limit was already used, check if it was
                    # already newly defined with ice free glacier
                    if was_errors[1]:
                        raise RuntimeError('Not able to minimise without ice '
                                           'free glacier after spinup! Best '
                                           'mismatch '
                                           f'{np.min(np.abs(mismatch))}%')
                    else:
                        # ok we set a new upper limit
                        t_spinup_limits[1] = (t_spinup_limits[1] +
                                              t_spinup_max_step_length)

            # now clip t_spinup with limits
            t_spinup = np.clip(t_spinup, t_spinup_limits[0], t_spinup_limits[1])

            # now start with mismatch calculation

            # if error during spinup (ice_free or out_of_domain) this defines
            # how much t_spinup is changed to look for an error free glacier spinup
            t_spinup_search_change = t_spinup_max_step_length / 10
            # maximum number of changes to look for an error free glacier
            max_iterations = int(t_spinup_max_step_length / t_spinup_search_change)
            is_out_of_domain = True
            is_ice_free_spinup = True
            is_ice_free_end = True
            is_first_guess_ice_free = False
            is_first_guess_out_of_domain = False
            doing_first_guess = (len(mismatch) == 0)
            define_new_lower_limit = False
            define_new_upper_limit = False
            iteration = 0

            while ((is_out_of_domain | is_ice_free_spinup | is_ice_free_end) &
                   (iteration < max_iterations)):
                try:
                    tmp_mismatch, is_ice_free_spinup = fct_to_minimise(t_spinup)

                    # no error occurred, so we are not outside the domain
                    is_out_of_domain = False

                    # check if we are ice_free after spinup, if so we search
                    # for a new upper limit for t_spinup
                    if is_ice_free_spinup:
                        was_errors[1] = True
                        define_new_upper_limit = True
                        # special treatment if it is the first guess
                        if np.isclose(t_spinup, first_guess_t_spinup) & \
                                doing_first_guess:
                            is_first_guess_ice_free = True
                            # here directly jump to the smaller limit
                            t_spinup = copy.copy(t_spinup_limits[0])
                        elif is_first_guess_ice_free & doing_first_guess:
                            # make large steps if it is first guess
                            t_spinup = t_spinup - t_spinup_max_step_length
                        else:
                            t_spinup = np.round(t_spinup - t_spinup_search_change,
                                                decimals=1)
                        if np.isclose(t_spinup, t_spinup_guess).any():
                            iteration = copy.copy(max_iterations)

                    # check if we are ice_free at the end of the model run, if
                    # so we use the lower t_spinup limit and change the limit if
                    # needed
                    elif np.isclose(tmp_mismatch, -100.):
                        is_ice_free_end = True
                        was_errors[1] = True
                        define_new_upper_limit = True
                        # special treatment if it is the first guess
                        if np.isclose(t_spinup, first_guess_t_spinup) & \
                                doing_first_guess:
                            is_first_guess_ice_free = True
                            # here directly jump to the smaller limit
                            t_spinup = copy.copy(t_spinup_limits[0])
                        elif is_first_guess_ice_free & doing_first_guess:
                            # make large steps if it is first guess
                            t_spinup = t_spinup - t_spinup_max_step_length
                        else:
                            # if lower limit was already used change it and use
                            if t_spinup == t_spinup_limits[0]:
                                t_spinup_limits[0] = (t_spinup_limits[0] -
                                                      t_spinup_max_step_length)
                                t_spinup = copy.copy(t_spinup_limits[0])
                            else:
                                # otherwise just try with a colder t_spinup
                                t_spinup = np.round(t_spinup - t_spinup_search_change,
                                                    decimals=1)

                    else:
                        is_ice_free_end = False

                except RuntimeError as e:
                    # check if glacier grow to large
                    if 'Glacier exceeds domain boundaries, at year:' in f'{e}':
                        # ok we where outside the domain, therefore we search
                        # for a new lower limit for t_spinup in 0.1 °C steps
                        is_out_of_domain = True
                        define_new_lower_limit = True
                        was_errors[0] = True
                        # special treatment if it is the first guess
                        if np.isclose(t_spinup, first_guess_t_spinup) & \
                                doing_first_guess:
                            is_first_guess_out_of_domain = True
                            # here directly jump to the larger limit
                            t_spinup = t_spinup_limits[1]
                        elif is_first_guess_out_of_domain & doing_first_guess:
                            # make large steps if it is first guess
                            t_spinup = t_spinup + t_spinup_max_step_length
                        else:
                            t_spinup = np.round(t_spinup + t_spinup_search_change,
                                                decimals=1)
                        if np.isclose(t_spinup, t_spinup_guess).any():
                            iteration = copy.copy(max_iterations)

                    else:
                        # otherwise this error can not be handled here
                        raise RuntimeError(e)

                iteration += 1

            if iteration >= max_iterations:
                # ok we were not able to find an mismatch without error
                # (ice_free or out of domain), so we try to raise an descriptive
                # RuntimeError
                if len(mismatch) == 0:
                    # unfortunately we were not able conduct one single error
                    # free run
                    msg = 'Not able to conduct one error free run. Error is '
                    if is_first_guess_ice_free:
                        msg += f'"ice_free" with last t_spinup of {t_spinup}.'
                    elif is_first_guess_out_of_domain:
                        msg += f'"out_of_domain" with last t_spinup of {t_spinup}.'
                    else:
                        raise RuntimeError('Something unexpected happened!')

                    raise RuntimeError(msg)

                elif define_new_lower_limit:
                    raise RuntimeError('Not able to minimise without '
                                       'exceeding the domain! Best '
                                       f'mismatch '
                                       f'{np.min(np.abs(mismatch))}%')
                elif define_new_upper_limit:
                    raise RuntimeError('Not able to minimise without ice '
                                       'free glacier after spinup! Best mismatch '
                                       f'{np.min(np.abs(mismatch))}%')
                elif is_ice_free_end:
                    raise RuntimeError('Not able to find a t_spinup so that '
                                       'glacier is not ice free at the end! '
                                       '(Last t_spinup '
                                       f'{t_spinup + t_spinup_max_step_length} °C)')
                else:
                    raise RuntimeError('Something unexpected happened during '
                                       'definition of new t_spinup limits!')
            else:
                # if we found a new limit set it
                if define_new_upper_limit & define_new_lower_limit:
                    # we can end here if we are at the first guess and took
                    # a to large step
                    was_errors[0] = False
                    was_errors[1] = False
                    if t_spinup <= t_spinup_limits[0]:
                        t_spinup_limits[0] = t_spinup
                        t_spinup_limits[1] = (t_spinup_limits[0] +
                                              t_spinup_max_step_length)
                    elif t_spinup >= t_spinup_limits[1]:
                        t_spinup_limits[1] = t_spinup
                        t_spinup_limits[0] = (t_spinup_limits[1] -
                                              t_spinup_max_step_length)
                    else:
                        if is_first_guess_ice_free:
                            t_spinup_limits[1] = t_spinup
                        elif is_out_of_domain:
                            t_spinup_limits[0] = t_spinup
                        else:
                            raise RuntimeError('I have not expected to get here!')
                elif define_new_lower_limit:
                    t_spinup_limits[0] = copy.copy(t_spinup)
                    if t_spinup >= t_spinup_limits[1]:
                        # this happens when the first guess was out of domain
                        was_errors[0] = False
                        t_spinup_limits[1] = (t_spinup_limits[0] +
                                              t_spinup_max_step_length)
                elif define_new_upper_limit:
                    t_spinup_limits[1] = copy.copy(t_spinup)
                    if t_spinup <= t_spinup_limits[0]:
                        # this happens when the first guess was ice free
                        was_errors[1] = False
                        t_spinup_limits[0] = (t_spinup_limits[1] -
                                              t_spinup_max_step_length)

            return float(tmp_mismatch), float(t_spinup)

        # first guess
        new_mismatch, new_t_spinup = get_mismatch(first_guess_t_spinup)
        t_spinup_guess.append(new_t_spinup)
        mismatch.append(new_mismatch)

        if abs(mismatch[-1]) < precision_percent:
            return t_spinup_guess, mismatch

        # second (arbitrary) guess is given depending on the outcome of first
        # guess, when mismatch is 100% t_spinup is changed for
        # t_spinup_max_step_length, but at least the second guess is 0.2 °C away
        # from the first guess
        step = np.sign(mismatch[-1]) * max(np.abs(mismatch[-1]) *
                                           t_spinup_max_step_length / 100,
                                           0.2)
        new_mismatch, new_t_spinup = get_mismatch(t_spinup_guess[0] + step)
        t_spinup_guess.append(new_t_spinup)
        mismatch.append(new_mismatch)

        if abs(mismatch[-1]) < precision_percent:
            return t_spinup_guess, mismatch

        # Now start with splin fit for guessing
        while len(t_spinup_guess) < maxiter:
            # get next guess from splin (fit partial linear function to previously
            # calculated (mismatch, t_spinup) pairs and get t_spinup value where
            # mismatch=0 from this fitted curve)
            sort_index = np.argsort(np.array(mismatch))
            tck = interpolate.splrep(np.array(mismatch)[sort_index],
                                     np.array(t_spinup_guess)[sort_index],
                                     k=1)
            # here we catch interpolation errors (two different t_spinup with
            # same mismatch), could happen if one t_spinup was close to a newly
            # defined limit
            if np.isnan(tck[1]).any():
                if was_errors[0]:
                    raise RuntimeError('Not able to minimise without '
                                       'exceeding the domain! Best '
                                       f'mismatch '
                                       f'{np.min(np.abs(mismatch))}%')
                elif was_errors[1]:
                    raise RuntimeError('Not able to minimise without ice '
                                       'free glacier! Best mismatch '
                                       f'{np.min(np.abs(mismatch))}%')
                else:
                    raise RuntimeError('Not able to minimise! Problem is '
                                       'unknown, need to check by hand! Best '
                                       'mismatch '
                                       f'{np.min(np.abs(mismatch))}%')
            new_mismatch, new_t_spinup = get_mismatch(float(interpolate.splev(0,
                                                                              tck)
                                                            ))
            t_spinup_guess.append(new_t_spinup)
            mismatch.append(new_mismatch)

            if abs(mismatch[-1]) < precision_percent:
                return t_spinup_guess, mismatch

        # Ok when we end here the spinup could not find satisfying match after
        # maxiter(ations)
        raise RuntimeError(f'Could not find mismatch smaller '
                           f'{precision_percent}% (only '
                           f'{np.min(np.abs(mismatch))}%) in {maxiter}'
                           f'Iterations!')

    # define function for the actual minimisation
    c_fun, model_dynamic_spinup_end, other_variable_mismatch = init_cost_fct()


    # define the MassBalanceModels for different spinup periods and try to
    # minimise, if minimisation fails a shorter spinup period is used
    # (first a spinup period between initial period and 'min_spinup_period'
    # years and the second try is to use a period of 'min_spinup_period' years,
    # if it still fails the actual error is raised)
    if spinup_start_yr is not None:
        spinup_period_initial = min(target_yr - spinup_start_yr,
                                    target_yr - yr_min)
    else:
        spinup_period_initial = min(spinup_period, target_yr - yr_min)
    if spinup_period_initial <= min_spinup_period:
        spinup_periods_to_try = [min_spinup_period]
    else:
        # try out a maximum of three different spinup_periods
        spinup_periods_to_try = [spinup_period_initial,
                                 int((spinup_period_initial +
                                      min_spinup_period) / 2),
                                 min_spinup_period]
    # after defining the initial spinup period we can define the year for the
    # fixed_geometry_spinup
    if add_fixed_geometry_spinup:
        fixed_geometry_spinup_yr = target_yr - spinup_period_initial
    else:
        fixed_geometry_spinup_yr = None

        # check if the user provided an mb_model_spinup, otherwise we must define a
        # new one each iteration
        provided_mb_model_spinup = False
        if mb_model_spinup is not None:
            provided_mb_model_spinup = True

        for spinup_period in spinup_periods_to_try:
            yr_spinup = target_yr - spinup_period

            if not provided_mb_model_spinup:
                # define spinup MassBalance
                # spinup is running for 'target_yr - yr_spinup' years, using a
                # ConstantMassBalance
                y0_spinup = (yr_spinup + target_yr) / 2
                halfsize_spinup = target_yr - y0_spinup
                mb_model_spinup = DistributedMassBalance(
                    gdir, mb_model_class=ConstantMassBalance,
                    filename=climate_filename,
                    input_filesuffix=climate_input_filesuffix,
                    y0=y0_spinup,
                    halfsize=halfsize_spinup,
                    use_distributed_data=True)

            # try to conduct minimisation, if an error occurred try shorter spinup
            # period
            try:
                final_t_spinup_guess, final_mismatch = minimise_with_spline_fit(c_fun)
                # ok no error occurred so we succeeded
                break
            except RuntimeError as e:
                # if the last spinup period was min_spinup_period the dynamic
                # spinup failed
                if spinup_period == min_spinup_period:
                    log.warning('No dynamic spinup could be conducted and the '
                                'original model with no spinup is saved using the '
                                f'provided output_filesuffix "{output_filesuffix}". '
                                f'The error message of the dynamic spinup is: {e}')
                    if ignore_errors:
                        model_dynamic_spinup_end = save_model_without_dynamic_spinup()
                        if return_t_spinup_best:
                            return model_dynamic_spinup_end, np.nan
                        else:
                            return model_dynamic_spinup_end
                    else:
                        # delete all files which could be saved during the previous
                        # iterations
                        if geom_path and os.path.exists(geom_path):
                            os.remove(geom_path)

                        # if fl_diag_path and os.path.exists(fl_diag_path):
                        #     os.remove(fl_diag_path)

                        if diag_path and os.path.exists(diag_path):
                            os.remove(diag_path)

                        raise RuntimeError(e)

        # hurray, dynamic spinup successfully
        gdir.add_to_diagnostics('run_dynamic_spinup_success', True)

        # also save some other stuff
        gdir.add_to_diagnostics('temp_bias_dynamic_spinup',
                                float(final_t_spinup_guess[-1]))
        gdir.add_to_diagnostics('dynamic_spinup_target_year',
                                int(target_yr))
        gdir.add_to_diagnostics('dynamic_spinup_period',
                                int(spinup_period))
        gdir.add_to_diagnostics('dynamic_spinup_forward_model_iterations',
                                int(forward_model_runs[-1]))
        gdir.add_to_diagnostics(f'{minimise_for}_mismatch_dynamic_spinup_{unit}_'
                                f'percent',
                                float(final_mismatch[-1]))
        gdir.add_to_diagnostics(f'reference_{minimise_for}_dynamic_spinup_{unit}',
                                float(reference_value))
        gdir.add_to_diagnostics('dynamic_spinup_other_variable_reference',
                                float(other_reference_value))
        gdir.add_to_diagnostics('dynamic_spinup_mismatch_other_variable_percent',
                                float(other_variable_mismatch[-1]))

        # here only save the final model state if store_model_evolution = False
        if not store_model_evolution:
            with warnings.catch_warnings():
                # For operational runs we ignore the warnings
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                model_dynamic_spinup_end[-1].run_until_and_store(
                    target_yr,
                    # geom_path=geom_path,
                    # diag_path=diag_path,
                    # fl_diag_path=fl_diag_path,
                    run_path = ioggm_diag_path
                    # store_monthly_step=store_monthly_step,
                )

        if return_t_spinup_best:
            return model_dynamic_spinup_end[-1], final_t_spinup_guess[-1]
        else:
            return model_dynamic_spinup_end[-1]