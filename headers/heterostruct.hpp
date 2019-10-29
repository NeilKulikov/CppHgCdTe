class heterostruct{
        private:
            double len = 0.;
            std::shared_ptr<staff::spline> spl = nullptr;
        public:
            heterostruct(
                std::vector<double> const & zs, 
                std::vector<double> const & xs){
                    const bool zsorted = std::is_sorted(zs.begin(), zs.end());
                    if(!zsorted)
                        throw std::logic_error("z should be sorted");
                    const auto zmm = std::minmax_element(zs.begin(), zs.end());
                    len = *zmm.second - *zmm.first;
                    auto zc = zs;
                    std::for_each(zc.begin(), zc.end(), [zmm](double & z){ z = z - *zmm.first; });
                    const auto xmm = std::minmax_element(xs.begin(), xs.end());
                    if(!((0. <= *xmm.first) && (*xmm.second <= 1.)))
                        throw std::domain_error("x should be in range [0;1]");
                    spl = std::make_shared<staff::spline>(zc, xs);
            };
            double length(void) const {
                return len;
            };
            double composition(double z) const {
                if(spl.get() == nullptr)
                    throw std::logic_error("Zero pointer spline in heterostruct");
                if(!((0. <= z) && (z <= len)))
                    throw std::out_of_range("z should be in interval [0;L]");
                return spl->eval(z);
            };
            model<double> at(const double z) const {
                const double comp = composition(z);
                auto rv = CdHgTe(comp);
                return rv;
            };
            
    };